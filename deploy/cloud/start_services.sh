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

# 2. Activate Python virtual environment or locate Python binary
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

PYTHON_BIN="$(which python3 || which python || echo 'python')"
if [ "$FOUND_VENV" -eq 0 ]; then
    echo "[INFO] No virtualenv activate script found. Using active Python: $PYTHON_BIN"
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
    echo "[INFO] Hugging Face token configured from environment."
elif [ -f "$HF_HOME/token" ]; then
    export HF_TOKEN="$(cat "$HF_HOME/token")"
    echo "[INFO] Loaded HF_TOKEN from $HF_HOME/token."
elif [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    echo "[INFO] Loaded HF_TOKEN from ~/.cache/huggingface/token."
elif [ -f "/workspace/.secrets/hf_token" ]; then
    export HF_TOKEN="$(cat "/workspace/.secrets/hf_token")"
    echo "[INFO] Loaded HF_TOKEN from /workspace/.secrets/hf_token."
elif [ -f ".secrets/hf_token" ]; then
    export HF_TOKEN="$(cat ".secrets/hf_token")"
    echo "[INFO] Loaded HF_TOKEN from .secrets/hf_token."
else
    echo "[WARNING] No HF_TOKEN detected! PersonaPlex-7B weights (nvidia/personaplex-7b-v1) require a Hugging Face token."
    echo "[WARNING] Set it with: export HF_TOKEN=\"hf_...\" before running this script if weights are not yet cached."
fi

# Modern PyTorch compatibility flag for Moshi
export NO_TORCH_COMPILE=1

# 4. Bootstrap Python Dependencies If Missing
if ! "$PYTHON_BIN" -c "import soundfile, fastapi, uvicorn, structlog" 2>/dev/null; then
    echo "[INFO] Gateway dependencies not detected. Installing project dependencies..."
    "$PYTHON_BIN" -m pip install -r requirements.txt
    "$PYTHON_BIN" -m pip install -e .
fi

# 5. Bootstrap Moshi (PersonaPlex Engine) If Missing
if ! "$PYTHON_BIN" -c "import moshi.server" 2>/dev/null; then
    echo "[INFO] Moshi engine not found in Python environment. Bootstrapping Moshi..."
    if [ -d "_personaplex_upstream/moshi" ]; then
        echo "[INFO] Installing Moshi from repository submodule (_personaplex_upstream/moshi)..."
        "$PYTHON_BIN" -m pip install rustymimi sounddevice einops sentencepiece accelerate || true
        "$PYTHON_BIN" -m pip install --no-deps -e "_personaplex_upstream/moshi" || true
    fi
    if ! "$PYTHON_BIN" -c "import moshi.server" 2>/dev/null; then
        PERSONAPLEX_DIR="$HOME/personaplex"
        if [ ! -d "$PERSONAPLEX_DIR" ]; then
            PERSONAPLEX_DIR="/workspace/personaplex"
        fi
        if [ ! -d "$PERSONAPLEX_DIR" ]; then
            echo "[INFO] Cloning https://github.com/NVIDIA/personaplex.git..."
            git clone https://github.com/NVIDIA/personaplex.git "$PERSONAPLEX_DIR" || true
        fi
        if [ -d "$PERSONAPLEX_DIR/moshi" ]; then
            if [ -f "$PERSONAPLEX_DIR/moshi/pyproject.toml" ]; then
                sed -i -E 's/torch<2.5,>=2.2.0/torch>=2.2.0/g; s/<2.5[0-9.]*//g' "$PERSONAPLEX_DIR/moshi/pyproject.toml" || true
            fi
            "$PYTHON_BIN" -m pip install rustymimi sounddevice einops sentencepiece accelerate || true
            "$PYTHON_BIN" -m pip install --no-deps -e "$PERSONAPLEX_DIR/moshi" || true
        fi
    fi
fi

# 6. Locate, Extract and Sync Voice Presets (.pt)
echo "[INFO] Verifying PersonaPlex voice presets (.pt)..."
"$PYTHON_BIN" scripts/prepare_voices.py || true

# 7. Apply PersonaPlex Upstream Server Patches
echo "[INFO] Applying PersonaPlex upstream stability patches..."
"$PYTHON_BIN" scripts/patch_moshi.py || true

VOICES_DIR=""
if [ -d "$HF_HOME/voices" ] && ls "$HF_HOME/voices"/*.pt >/dev/null 2>&1; then
    VOICES_DIR="$HF_HOME/voices"
elif [ -d "voices" ] && ls "voices"/*.pt >/dev/null 2>&1; then
    VOICES_DIR="$(pwd)/voices"
elif [ -d "/workspace/voices" ] && ls "/workspace/voices"/*.pt >/dev/null 2>&1; then
    VOICES_DIR="/workspace/voices"
else
    mkdir -p voices
    VOICES_DIR="$(pwd)/voices"
fi

echo "=============================================================================="
echo " Starting PersonaPlex 7B Voice Services on NVIDIA A100 GPU"
echo " HF_HOME:    ${HF_HOME}"
echo " Voices Dir: ${VOICES_DIR}"
echo " Python:     ${PYTHON_BIN}"
echo "=============================================================================="

# 8. Start Moshi 7B Inference Worker
mkdir -p logs
echo "[INFO] Launching PersonaPlex 7B Worker on 127.0.0.1:8998 (logging to logs/worker.log)..."
WORKER_CMD="$PYTHON_BIN -u -m moshi.server --host 127.0.0.1 --port 8998 --voice-prompt-dir $VOICES_DIR"

$WORKER_CMD 2>&1 | tee -a logs/worker.log worker.log &
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

# 9. Launch Orchestration Gateway
echo ""
echo "=============================================================================="
if [ "$WORKER_READY" -eq 1 ]; then
    echo " 🚀 PersonaPlex 7B GPU Mode Active (A100 SXM4)"
    echo " ✨ CONSOLE: Click 'HTTP Service [Port 8000]' in Krutrim Cloud"
    echo "=============================================================================="
    "$PYTHON_BIN" -m orchestration.cli run-gateway \
        --host 0.0.0.0 \
        --port 8000 \
        --worker 127.0.0.1:8998 \
        --worker-type personaplex
else
    echo " ⚠️ Falling back to Local Voice Engine..."
    echo " ✨ CONSOLE: Click 'HTTP Service [Port 8000]' in Krutrim Cloud"
    echo "=============================================================================="
    "$PYTHON_BIN" -m orchestration.cli run-gateway \
        --host 0.0.0.0 \
        --port 8000 \
        --worker-type mock
fi
