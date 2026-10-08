"""
Standalone script to patch upstream moshi.server with PersonaPlex stability fixes.
Safe to run idempotently. Does not throw if moshi is not yet installed.
"""

import sys
from pathlib import Path


def patch_moshi_server(target_path: str | Path | None = None) -> bool:
    if target_path:
        server_path = Path(target_path)
    else:
        try:
            import moshi.server
            server_path = Path(moshi.server.__file__)
        except ImportError:
            print("[INFO] moshi module not found in current Python environment. Skipping patch.")
            return False

    if not server_path.exists():
        print(f"[WARNING] moshi.server not found at {server_path}")
        return False

    with open(server_path, "r", encoding="utf-8") as f:
        src = f.read()

    modified = False

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
                pt_candidates = sorted(glob.glob(os.path.join(self.voice_prompt_dir, "*.pt")))
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
