"""
Standalone script to patch upstream moshi.server with PersonaPlex stability fixes.
Safe to run idempotently.
Does NOT import moshi.server directly to prevent top-level main() execution.
"""

import importlib.util
import os
import sys
from pathlib import Path


def find_moshi_server_file() -> Path | None:
    # 1. Search common deployment paths
    candidates = [
        Path("/workspace/personaplex/moshi/moshi/server.py"),
        Path.home() / "personaplex/moshi/moshi/server.py",
        Path("_personaplex_upstream/moshi/moshi/server.py"),
        Path("./_personaplex_upstream/moshi/moshi/server.py"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    # 2. Inspect module spec without executing module
    try:
        spec = importlib.util.find_spec("moshi.server")
        if spec and spec.origin and Path(spec.origin).exists():
            return Path(spec.origin).resolve()
    except Exception:
        pass

    # 3. Inspect parent moshi package spec
    try:
        spec = importlib.util.find_spec("moshi")
        if spec and spec.origin:
            p = Path(spec.origin).parent / "server.py"
            if p.exists():
                return p.resolve()
    except Exception:
        pass

    return None


def patch_moshi_server(target_path: str | Path | None = None) -> bool:
    if target_path:
        server_path = Path(target_path)
    else:
        server_path = find_moshi_server_file()

    if not server_path or not server_path.exists():
        print(f"[INFO] moshi.server not found at target ({server_path}). Skipping patch.")
        return False

    with open(server_path, "r", encoding="utf-8") as f:
        src = f.read()

    modified = False

    # Fix 0: Wrap top-level main() call with `if __name__ == '__main__':`
    # Upstream moshi/server.py puts `with torch.no_grad(): main()` at module root,
    # causing any import of moshi.server to execute main() immediately.
    if 'if __name__ == "__main__":' not in src:
        if "with torch.no_grad():\n    main()" in src:
            src = src.replace(
                "with torch.no_grad():\n    main()",
                'if __name__ == "__main__":\n    with torch.no_grad():\n        main()',
            )
            modified = True
        elif "main()" in src:
            src = src.replace(
                "main()",
                'if __name__ == "__main__":\n    main()',
            )
            modified = True

    # Fix 1: Seed parameter query dictionary lookup
    if 'request["seed"]' in src:
        src = src.replace('request["seed"]', 'request.query["seed"]')
        modified = True

    # Fix 2: voice_prompt_path NoneType guard before endswith check
    old_vp_block = "if self.lm_gen.voice_prompt != voice_prompt_path:\n            if voice_prompt_path.endswith('.pt'):"
    new_vp_block = "if voice_prompt_path is not None and self.lm_gen.voice_prompt != voice_prompt_path:\n            if voice_prompt_path.endswith('.pt'):"
    if old_vp_block in src:
        src = src.replace(old_vp_block, new_vp_block)
        modified = True

    # Fix 3: Fallback on missing voice file instead of crashing TCP session
    old_fnf = "raise FileNotFoundError(\n                    f\"Requested voice prompt '{voice_prompt_filename}' not found in '{self.voice_prompt_dir}'\"\n                )"
    new_fnf = """import glob
                pt_candidates = sorted(glob.glob(os.path.join(self.voice_prompt_dir, "*.pt"))) + sorted(glob.glob(os.path.join(self.voice_prompt_dir, "*.wav")))
                if pt_candidates:
                    voice_prompt_path = pt_candidates[0]
                    clog.log("warning", f"Requested voice '{voice_prompt_filename}' not found, falling back to {voice_prompt_path}")
                else:
                    voice_prompt_path = None"""
    if old_fnf in src:
        src = src.replace(old_fnf, new_fnf)
        modified = True

    # Fix 4: Log crashes and close diagnostics in worker task loops
    old_wait = "done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)\n                # Force-kill remaining tasks"
    new_wait = """done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if task.exception():
                        import traceback
                        clog.log("error", f"CRASH in worker task {task}: {task.exception()}")
                        clog.log("error", "".join(traceback.format_exception(type(task.exception()), task.exception(), task.exception().__traceback__)))
                    else:
                        clog.log("info", f"Worker task finished normally: {task}")
                # Force-kill remaining tasks"""
    if old_wait in src:
        src = src.replace(old_wait, new_wait)
        modified = True

    # Fix 5: Log client close frame codes in recv_loop
    old_close = "elif message.type == aiohttp.WSMsgType.CLOSED:\n                        break\n                    elif message.type == aiohttp.WSMsgType.CLOSE:\n                        break"
    new_close = 'elif message.type == aiohttp.WSMsgType.CLOSED:\n                        clog.log("info", f"ws CLOSED by client (code={ws.close_code})")\n                        break\n                    elif message.type == aiohttp.WSMsgType.CLOSE:\n                        clog.log("info", f"ws CLOSE received from client (code={ws.close_code})")\n                        break'
    if old_close in src:
        src = src.replace(old_close, new_close)
        modified = True

    # Fix 6: Protect ws.send_bytes with send_lock and protect opus_reader append against corrupt packet crashes
    if "send_lock = asyncio.Lock()" not in src:
        src = src.replace(
            "opus_reader.append_bytes(payload)",
            'try:\n                            opus_reader.append_bytes(payload)\n                        except Exception as e:\n                            clog.log("warning", f"opus_reader decode warning: {e}")',
        )
        src = src.replace(
            'clog.log("info", "connection closed")',
            'clog.log("info", "connection closed")\n\n        send_lock = asyncio.Lock()',
        )
        src = src.replace(
            "await ws.send_bytes(msg)",
            "async with send_lock:\n                                await ws.send_bytes(msg)",
        )
        src = src.replace(
            'await ws.send_bytes(b"\\x01" + msg)',
            'async with send_lock:\n                        await ws.send_bytes(b"\\x01" + msg)',
        )
        modified = True

    # Fix 7: Bypass dist.tgz static download (Gateway on port 8000 serves UI, worker does not need static assets)
    if 'dist_tgz = hf_hub_download("nvidia/personaplex-7b-v1", "dist.tgz")' in src:
        src = src.replace(
            'dist_tgz = hf_hub_download("nvidia/personaplex-7b-v1", "dist.tgz")',
            'return None  # Bypass dist.tgz; Gateway serves console',
        )
        modified = True

    # Fix 8: Forward HF_TOKEN to hf_hub_download automatically
    hf_token_patch_marker = "# HF_TOKEN auto-inject wrapper"
    if hf_token_patch_marker not in src and "from huggingface_hub import hf_hub_download" in src:
        hf_replacement = """from huggingface_hub import hf_hub_download as _raw_hf_hub_download
# HF_TOKEN auto-inject wrapper
def hf_hub_download(*args, **kwargs):
    if "token" not in kwargs:
        token_env = os.environ.get("HF_TOKEN")
        if token_env:
            kwargs["token"] = token_env
    return _raw_hf_hub_download(*args, **kwargs)
"""
        src = src.replace("from huggingface_hub import hf_hub_download", hf_replacement, 1)
        modified = True

    if modified:
        with open(server_path, "w", encoding="utf-8") as f:
            f.write(src)
        print(f"[SUCCESS] Patched moshi.server at {server_path}")
        return True
    else:
        print(f"[INFO] moshi.server already patched or up to date at {server_path}")
        return False


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    patch_moshi_server(target)
