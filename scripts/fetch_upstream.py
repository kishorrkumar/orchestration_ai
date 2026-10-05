#!/usr/bin/env python3
"""
Reproducible fetch script for the upstream NVIDIA PersonaPlex reference repository.
Pins to verified upstream commit: 3428dfd95309a7f3c84fd93259ded0f810d1ff91
"""

import os
import subprocess
import sys
from pathlib import Path

UPSTREAM_REPO = "https://github.com/NVIDIA/personaplex.git"
PINNED_COMMIT = "3428dfd95309a7f3c84fd93259ded0f810d1ff91"
TARGET_DIR = Path(__file__).resolve().parent.parent / "_personaplex_upstream"


def main() -> int:
    print(f"--> Target upstream directory: {TARGET_DIR}")
    if not TARGET_DIR.exists():
        print(f"--> Cloning {UPSTREAM_REPO} into {TARGET_DIR}...")
        subprocess.check_call(["git", "clone", "--filter=blob:none", UPSTREAM_REPO, str(TARGET_DIR)])
    else:
        print(f"--> Fetching latest references for {TARGET_DIR}...")
        subprocess.check_call(["git", "-C", str(TARGET_DIR), "fetch", "origin"])

    print(f"--> Checking out pinned commit: {PINNED_COMMIT}...")
    subprocess.check_call(["git", "-C", str(TARGET_DIR), "checkout", PINNED_COMMIT])

    actual_commit = subprocess.check_output(
        ["git", "-C", str(TARGET_DIR), "rev-parse", "HEAD"], text=True
    ).strip()

    if actual_commit != PINNED_COMMIT:
        print(f"ERROR: Expected commit {PINNED_COMMIT} but got {actual_commit}", file=sys.stderr)
        return 1

    print(f"[OK] Upstream PersonaPlex verified at commit {actual_commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
