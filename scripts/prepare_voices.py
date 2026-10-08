"""
Voice presets preparation utility for PersonaPlex 7B.
Discovers, extracts, and syncs .pt voice embeddings between HF cache and local voices dir.
"""

import os
import shutil
import tarfile
from pathlib import Path


def prepare_voices() -> int:
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    token = os.environ.get("HF_TOKEN")
    voices_dir = Path(hf_home) / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)

    # 1. Search specific targeted directories for voice presets
    candidate_paths = [
        voices_dir,
        Path("voices").resolve(),
        Path("data/cloned_voices").resolve(),
        Path(hf_home).resolve() / "hub",
        Path("/workspace/voices").resolve() if Path("/workspace/voices").exists() else None,
        Path("/workspace/huggingface/voices").resolve() if Path("/workspace/huggingface/voices").exists() else None,
        Path("/home/jovyan/.cache/huggingface/voices").resolve() if Path("/home/jovyan/.cache/huggingface/voices").exists() else None,
    ]

    for cpath in candidate_paths:
        if cpath and cpath.exists():
            try:
                for pt in cpath.glob("*.pt"):
                    dest = voices_dir / pt.name
                    if not dest.exists():
                        try:
                            shutil.copy2(str(pt), str(dest))
                        except Exception:
                            pass
            except Exception:
                pass

    # 2. If no presets found, check for cached voices.tgz in HF cache
    existing = list(voices_dir.glob("*.pt"))
    if len(existing) == 0:
        tgz_candidates = [
            Path(hf_home) / "voices.tgz",
            Path("voices.tgz"),
            Path("/workspace/voices.tgz"),
        ]
        found_tgz = None
        for tgz in tgz_candidates:
            if tgz.exists():
                found_tgz = tgz
                break

        if found_tgz:
            try:
                print(f"[INFO] Extracting cached archive: {found_tgz}")
                with tarfile.open(found_tgz, "r:gz") as tar:
                    kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
                    tar.extractall(path=voices_dir, **kwargs)
            except Exception as e:
                print(f"[INFO] Cached tar extraction note: {e}")
        elif token:
            try:
                from huggingface_hub import hf_hub_download
                print("[INFO] Attempting download of voices.tgz from nvidia/personaplex-7b-v1...")
                downloaded = hf_hub_download("nvidia/personaplex-7b-v1", "voices.tgz", token=token)
                with tarfile.open(downloaded, "r:gz") as tar:
                    kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
                    tar.extractall(path=voices_dir, **kwargs)
            except Exception as e:
                print(f"[INFO] Archive download/extraction note: {e}")

    # 3. Flatten any nested extracted folders (voices.tgz extracts a 'voices/' directory)
    for pt in list(voices_dir.rglob("*.pt")):
        if pt.parent != voices_dir:
            dest = voices_dir / pt.name
            if not dest.exists():
                try:
                    shutil.move(str(pt), str(dest))
                except Exception:
                    pass

    # 4. Sync to local ./voices directory
    proj_voices = Path("voices")
    proj_voices.mkdir(exist_ok=True)
    for pt in voices_dir.glob("*.pt"):
        dest = proj_voices / pt.name
        if not dest.exists():
            try:
                dest.symlink_to(pt)
            except Exception:
                try:
                    shutil.copy2(str(pt), str(dest))
                except Exception:
                    pass

    found_presets = sorted([p.name for p in voices_dir.glob("*.pt")])
    print(f"[INFO] Voice presets verified: {len(found_presets)} presets available in {voices_dir}")
    if len(found_presets) == 0:
        print(f"[WARNING] No voice preset .pt files found in {voices_dir}; worker will use fallback or download on demand.")
    return len(found_presets)


if __name__ == "__main__":
    prepare_voices()
