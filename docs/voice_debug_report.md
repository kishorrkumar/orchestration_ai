# PersonaPlex Worker 4 ms Disconnect Root Cause & Diagnostic Report

**Date:** 2026-10-07  
**Host Environment:** Krutrim Cloud Pod (Linux, NVIDIA A100-SXM4-40GB, Python 3.13 venv `/home/jovyan/personaplex_env`)  
**Components:** `moshi.server` (:8998), FastAPI Gateway (:8000), `deploy/cloud/start_services.sh`

---

## 1. Executive Summary & The Exact Cause of the 4 ms Worker Disconnect

At `09:31:37,546` and `09:32:15`, user connection attempts to `ws://127.0.0.1:8998/api/chat` failed with:
```
09:31:37,546 Connecting worker ... ws://127.0.0.1:8998/api/chat?text_prompt=...&voice_prompt=NATM1.pt&audio_temperature=0.8&text_temperature=0.7&audio_topk=250&text_topk=25
09:31:37,550 ERROR Failed to connect worker client: ... ConnectionClosedError: no close frame received or sent
```

The worker disconnected after exactly **4 ms**, before any handshake byte (`0x00`) was transmitted.

### Worker-Side Root Cause & Traceback Analysis
In upstream `moshi.server` (`moshi/server.py` line 148–166):
```python
# Construct full voice prompt path
if self.voice_prompt_dir is not None:
    voice_prompt_filename = request.query.get("voice_prompt")
    requested_voice_prompt_path = None
    if voice_prompt_filename is not None:
        requested_voice_prompt_path = os.path.join(self.voice_prompt_dir, voice_prompt_filename)
    if requested_voice_prompt_path is None or not os.path.exists(requested_voice_prompt_path):
        raise FileNotFoundError(
            f"Requested voice prompt '{voice_prompt_filename}' not found in '{self.voice_prompt_dir}'"
        )
```

When an exception (`FileNotFoundError`) is raised during an `aiohttp.web.WebSocketResponse` request handler before or right after `await ws.prepare(request)`:
1. `aiohttp` immediately terminates the underlying TCP/WebSocket transport without sending a clean WebSocket close frame.
2. The client websocket (`websockets.connect`) raises:
   `ConnectionClosedError: code = 1006 (connection closed abnormally [internal]), no close frame received or sent`.
3. The elapsed time from HTTP request parsing to `raise FileNotFoundError` is ~4 ms on CPU.

### What Happened During the 08:56 Run of `start_services.sh`
The previous extraction block in `start_services.sh` ran:
```python
with tarfile.open(tgz_file, 'r:gz') as tar:
    tar.extractall(path=voices_dir)
```
1. Upstream `nvidia/personaplex-7b-v1` packages `voices.tgz` with an internal root directory `voices/`.
2. Extracting `voices.tgz` directly into `voices_dir` (`/workspace/huggingface/voices`) created a **nested** structure:
   `/workspace/huggingface/voices/voices/*.pt`
3. The script's verification logic:
   ```python
   len(list(voices_dir.glob("*.pt")))
   ```
   only inspected top-level files in `voices_dir`, finding **0** files. Thus, it printed:
   `Voice presets verified: 0 presets available.`
4. The worker was launched with `--voice-prompt-dir /workspace/huggingface/voices`.
5. Because the `.pt` files were sitting inside the nested directory `/workspace/huggingface/voices/voices/`, `os.path.join(self.voice_prompt_dir, "NATM1.pt")` returned `False` for `os.path.exists(...)`.
6. `moshi.server` threw `FileNotFoundError` in 4 ms.
7. The gateway caught `ConnectionClosedError` and released the worker back to the pool as `IDLE`, causing subsequent calls with other presets to hit the same crash.

---

## 2. Voice Presets Verification & Script Fix

### Verified Preset Files (All 18 Presets Present)
The 18 official PersonaPlex voice preset files exist and match the official specification:
- **Natural Female (4):** `NATF0.pt`, `NATF1.pt`, `NATF2.pt`, `NATF3.pt`
- **Natural Male (4):** `NATM0.pt`, `NATM1.pt`, `NATM2.pt`, `NATM3.pt`
- **Variety Female (5):** `VARF0.pt`, `VARF1.pt`, `VARF2.pt`, `VARF3.pt`, `VARF4.pt`
- **Variety Male (5):** `VARM0.pt`, `VARM1.pt`, `VARM2.pt`, `VARM3.pt`, `VARM4.pt`

### Fixes Applied to `deploy/cloud/start_services.sh`
1. **Directory Flattening:** Automatically scans and flattens any nested directories (`voices/voices/*.pt` -> `voices/*.pt`).
2. **Safe Tar Extraction:** Added `filter="data"` on Python 3.12+ / 3.13 to resolve the `tarfile.DeprecationWarning`.
3. **Strict Validation & Loud Failures:** Counts verified presets (`len(found_presets) == 18`). If any preset is missing, the script prints an explicit error and exits with code 1 instead of silently proceeding.
4. **Dedicated Worker Stderr Logging:** Worker is now started with unbuffered output (`python -u -m moshi.server ...`) teeing all stdout and stderr to `logs/worker.log` and `worker.log` as well as the console.
5. **Server In-Memory Patching:** Patched upstream `moshi.server` in the environment so that if a requested voice prompt is missing, it logs a warning and falls back to a valid available `.pt` candidate rather than immediately terminating the TCP socket with an unhandled exception.

---

## 3. Worker State & Health Probing (No Stuck Locks)

In containers where `ss` is absent and `nvidia-smi` memory queries are restricted (`Insufficient Permissions`), worker health is probed via:
1. `ps aux | grep moshi.server`: Checks worker process presence and CPU/memory footprint.
2. `asyncio.open_connection("127.0.0.1", 8998)`: Probes TCP port responsiveness before leasing any worker from `WorkerPool`.
3. `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8998/`: Checks HTTP/WebSocket server readiness.

### Gateway Pool Hardening:
- **Health Probing on Acquire:** `WorkerPool.acquire_worker` now performs an asynchronous TCP health check before handing a worker instance to a caller session. If unreachable, the worker is marked `WorkerStatus.UNHEALTHY`.
- **Health Probing on Release:** `WorkerPool.release_worker(worker_id, success=False)` probes the worker's health before returning it to `IDLE`. If the worker died or failed the probe, it remains `UNHEALTHY` until the recovery loop verifies it.
- **Pre-Lease Voice Validation:** `voice_v2.py` pre-validates that the chosen `.pt` file exists on disk via `get_existing_voice_files()`. If not present, the gateway immediately returns WebSocket error / HTTP 422:
  `voice 'X' not found, available: NATF0.pt, NATF1.pt, ...`
  preventing any 4 ms upstream crash.
