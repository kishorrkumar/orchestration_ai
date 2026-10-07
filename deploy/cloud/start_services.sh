#!/usr/bin/env bash
# ==============================================================================
# Robust PersonaPlex 7B & Orchestration Gateway Launcher for Krutrim Cloud
# ==============================================================================

set -euo pipefail

# 1. Clean up any stale processes on ports 8000 and 8998
echo "[INFO] Freeing ports 8000 and 8998..."
pkill -f "moshi.server" 2>/dev/null || true
pkill -f "orchestration.cli" 2>/dev/null || true
kill -9 $(lsof -t -i:8000 2>/dev/null) 2>/dev/null || true
kill -9 $(lsof -t -i:8998 2>/dev/null) 2>/dev/null || true
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 8998/tcp 2>/dev/null || true

# 2. Activate Python virtual environment
FOUND_VENV=0
for venv_candidate in \
    "$HOME/personaplex_env" \
    "/workspace/personaplex_env" \
    "./.venv" \
    "$HOME/.venv" \
    "/workspace/.venv" \
    "$HOME/venv" \
    "/workspace/venv"; do
    if [ -f "$venv_candidate/bin/activate" ]; then
        echo "[INFO] Activating virtual environment: $venv_candidate"
        # shellcheck disable=SC1090
        source "$venv_candidate/bin/activate"
        FOUND_VENV=1
        break
    fi
done

if [ "$FOUND_VENV" -eq 0 ]; then
    echo "[INFO] No virtualenv activate script found. Using system python3: $(which python3 || echo 'none')"
fi

# 3. Configure Persistent HF Cache & Token
if [ -d "/workspace/huggingface" ]; then
    export HF_HOME="/workspace/huggingface"
elif [ -d "/data/huggingface" ]; then
    export HF_HOME="/data/huggingface"
else
    export HF_HOME="$HOME/.cache/huggingface"
fi

if [ -n "${HF_TOKEN:-}" ]; then
    mkdir -p ~/.cache/huggingface "$HF_HOME"
    echo -n "$HF_TOKEN" > ~/.cache/huggingface/token
    echo -n "$HF_TOKEN" > "$HF_HOME/token"
    echo "[INFO] Hugging Face token configured."
elif [ -f "$HF_HOME/token" ]; then
    export HF_TOKEN="$(cat "$HF_HOME/token")"
    echo "[INFO] Loaded HF_TOKEN from $HF_HOME/token."
elif [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    echo "[INFO] Loaded HF_TOKEN from ~/.cache/huggingface/token."
else
    echo "[WARNING] No HF_TOKEN detected! PersonaPlex-7B weights (nvidia/personaplex-7b-v1) require a Hugging Face token."
    echo "[WARNING] Set it with: export HF_TOKEN=\"hf_...\" before running this script."
fi

# Modern PyTorch compatibility flag for Moshi
export NO_TORCH_COMPILE=1

# 4. Locate and Extract Voice Presets (.pt)
echo "[INFO] Verifying PersonaPlex voice presets (.pt)..."
python3 -c "
import os, tarfile, shutil
from pathlib import Path

hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
token = os.environ.get('HF_TOKEN')
voices_dir = Path(hf_home) / 'voices'
voices_dir.mkdir(parents=True, exist_ok=True)

pt_files = list(voices_dir.glob('*.pt'))
if len(pt_files) < 18:
    print('[INFO] Extracting/downloading 18 PersonaPlex voice presets...')
    tgz_candidates = list(Path(hf_home).glob('**/voices.tgz'))
    if tgz_candidates:
        tgz_file = tgz_candidates[0]
        print(f'[INFO] Extracting cached archive: {tgz_file}')
        with tarfile.open(tgz_file, 'r:gz') as tar:
            tar.extractall(path=voices_dir)
    else:
        try:
            from huggingface_hub import hf_hub_download
            print('[INFO] Downloading voices.tgz from nvidia/personaplex-7b-v1...')
            downloaded = hf_hub_download('nvidia/personaplex-7b-v1', 'voices.tgz', token=token)
            with tarfile.open(downloaded, 'r:gz') as tar:
                tar.extractall(path=voices_dir)
        except Exception as e:
            print(f'[WARNING] Could not fetch voices.tgz: {e}')

# Sync to local ./voices directory
proj_voices = Path('voices')
proj_voices.mkdir(exist_ok=True)
for pt in voices_dir.glob('*.pt'):
    dest = proj_voices / pt.name
    if not dest.exists():
        try:
            dest.symlink_to(pt)
        except Exception:
            shutil.copy2(pt, dest)

print(f'[INFO] Voice presets verified: {len(list(voices_dir.glob(\"*.pt\")))} presets available.')
" || true

# 5. Patch moshi.server upstream bugs in active environment
python3 -c "
try:
    import moshi.server
    server_path = moshi.server.__file__
    with open(server_path, 'r', encoding='utf-8') as f:
        src = f.read()

    modified = False
    # Fix 1: seed KeyError
    if 'request[\"seed\"]' in src:
        src = src.replace('request[\"seed\"]', 'request.query[\"seed\"]')
        modified = True

    # Fix 2: voice_prompt_path NoneType endswith
    old_block = 'if self.lm_gen.voice_prompt != voice_prompt_path:\n            if voice_prompt_path.endswith(\'.pt\'):'
    new_block = 'if voice_prompt_path is not None and self.lm_gen.voice_prompt != voice_prompt_path:\n            if voice_prompt_path.endswith(\'.pt\'):'
    if old_block in src:
        src = src.replace(old_block, new_block)
        modified = True

    # Fix 3: Fallback on missing voice file instead of crashing TCP
    old_fnf = 'raise FileNotFoundError(\n                    f\"Requested voice prompt \'{voice_prompt_filename}\' not found in \'{self.voice_prompt_dir}\"\"\n                )'
    new_fnf = '''import glob
                pt_candidates = sorted(glob.glob(os.path.join(self.voice_prompt_dir, \"*.pt\")))
                if pt_candidates:
                    voice_prompt_path = pt_candidates[0]
                    clog.log(\"warning\", f\"Requested voice \'{voice_prompt_filename}\' not found, falling back to {voice_prompt_path}\")
                else:
                    voice_prompt_path = None'''
    if old_fnf in src:
        src = src.replace(old_fnf, new_fnf)
        modified = True

    # Fix 4: Log crashes and close diagnostics in worker task loops
    old_wait = 'done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)\n                # Force-kill remaining tasks'
    new_wait = '''done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if task.exception():
                        import traceback
                        clog.log(\"error\", f\"CRASH in worker task {task}: {task.exception()}\")
                        clog.log(\"error\", \"\".join(traceback.format_exception(type(task.exception()), task.exception(), task.exception().__traceback__)))
                    else:
                        clog.log(\"info\", f\"Worker task finished normally: {task}\")
                # Force-kill remaining tasks'''
    if old_wait in src:
        src = src.replace(old_wait, new_wait)
        modified = True

    # Fix 5: Log client close frame codes in recv_loop
    old_close = 'elif message.type == aiohttp.WSMsgType.CLOSED:\n                        break\n                    elif message.type == aiohttp.WSMsgType.CLOSE:\n                        break'
    new_close = 'elif message.type == aiohttp.WSMsgType.CLOSED:\n                        clog.log(\"info\", f\"ws CLOSED by client (code={ws.close_code})\")\n                        break\n                    elif message.type == aiohttp.WSMsgType.CLOSE:\n                        clog.log(\"info\", f\"ws CLOSE received from client (code={ws.close_code})\")\n                        break'
    if old_close in src:
        src = src.replace(old_close, new_close)
        modified = True

    # Fix 6: Protect ws.send_bytes with send_lock and protect opus_reader append against corrupt packet crashes
    if 'send_lock = asyncio.Lock()' not in src:
        src = src.replace(
            'opus_reader.append_bytes(payload)',
            'try:\n                            opus_reader.append_bytes(payload)\n                        except Exception as e:\n                            clog.log(\"warning\", f\"opus_reader decode warning: {e}\")'
        )
        src = src.replace(
            'clog.log("info", "connection closed")',
            'clog.log("info", "connection closed")\n\n        send_lock = asyncio.Lock()'
        )
        src = src.replace(
            'await ws.send_bytes(msg)',
            'async with send_lock:\n                                await ws.send_bytes(msg)'
        )
        src = src.replace(
            'await ws.send_bytes(b"\\x01" + msg)',
            'async with send_lock:\n                        await ws.send_bytes(b"\\x01" + msg)'
        )
        modified = True

    if modified:
        with open(server_path, 'w', encoding='utf-8') as f:
            f.write(src)
        print(f'[SUCCESS] Patched moshi.server at {server_path}')
    else:
        print('[INFO] moshi.server already patched.')
except Exception as patch_err:
    print(f'[INFO] moshi patch status: {patch_err}')
" || true

VOICES_DIR=""
if [ -d "$HF_HOME/voices" ] && [ -n "$(ls -A "$HF_HOME/voices"/*.pt 2>/dev/null)" ]; then
    VOICES_DIR="$HF_HOME/voices"
elif [ -d "voices" ] && [ -n "$(ls -A "voices"/*.pt 2>/dev/null)" ]; then
    VOICES_DIR="$(pwd)/voices"
else
    mkdir -p voices
    VOICES_DIR="$(pwd)/voices"
fi

echo "=============================================================================="
echo " Starting PersonaPlex 7B Voice Services on NVIDIA A100 GPU"
echo " HF_HOME:    ${HF_HOME}"
echo " Voices Dir: ${VOICES_DIR}"
echo "=============================================================================="

# 6. Start Moshi 7B Inference Worker
echo "[INFO] Launching PersonaPlex 7B Worker on 127.0.0.1:8998..."
WORKER_CMD="python -m moshi.server --host 127.0.0.1 --port 8998 --voice-prompt-dir $VOICES_DIR"

$WORKER_CMD > worker.log 2>&1 &
WORKER_PID=$!
echo "[INFO] Worker spawned (PID: $WORKER_PID). Waiting for 14.5 GB weights to load into A100 VRAM..."

WORKER_READY=0
for i in $(seq 1 45); do
    if timeout 1 bash -c "</dev/tcp/127.0.0.1/8998" 2>/dev/null; then
        echo ""
        echo "[SUCCESS] PersonaPlex 7B GPU Worker is ONLINE and accepting connections!"
        WORKER_READY=1
        break
    fi
    # Check if process died
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
        echo ""
        echo "[WARNING] Worker process exited early. Checking worker.log..."
        tail -n 25 worker.log 2>/dev/null || true
        break
    fi
    echo -n "."
    sleep 1
done

# 6. Launch Orchestration Gateway
echo ""
echo "=============================================================================="
if [ "$WORKER_READY" -eq 1 ]; then
    echo " 🚀 PersonaPlex 7B GPU Mode Active (A100 SXM4 40GB)"
    echo " ✨ CONSOLE: Click 'HTTP Service [Port 8000]' in Krutrim Cloud"
    echo "=============================================================================="
    python -m orchestration.cli run-gateway \
        --host 0.0.0.0 \
        --port 8000 \
        --worker 127.0.0.1:8998 \
        --worker-type personaplex
else
    echo " ⚠️ Falling back to Local Voice Engine..."
    echo " ✨ CONSOLE: Click 'HTTP Service [Port 8000]' in Krutrim Cloud"
    echo "=============================================================================="
    python -m orchestration.cli run-gateway \
        --host 0.0.0.0 \
        --port 8000 \
        --worker-type mock
fi
