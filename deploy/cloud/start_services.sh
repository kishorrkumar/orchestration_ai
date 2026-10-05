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
    mkdir -p ~/.cache/huggingface
    echo -n "$HF_TOKEN" > ~/.cache/huggingface/token
fi

# Modern PyTorch compatibility flag for Moshi
export NO_TORCH_COMPILE=1

# 4. Locate Voice Presets
VOICES_DIR=""
if [ -d "$HF_HOME/voices" ]; then
    VOICES_DIR="$HF_HOME/voices"
else
    FOUND=$(find "$HF_HOME" -type d -name "voices" 2>/dev/null | head -n 1)
    if [ -n "$FOUND" ]; then
        VOICES_DIR="$FOUND"
    fi
fi

echo "=============================================================================="
echo " Starting PersonaPlex 7B Voice Services on NVIDIA A100 GPU"
echo " HF_HOME:    ${HF_HOME}"
echo " Voices Dir: ${VOICES_DIR:-none}"
echo "=============================================================================="

# 5. Start Moshi 7B Inference Worker
echo "[INFO] Launching PersonaPlex 7B Worker on 127.0.0.1:8998..."
WORKER_CMD="python -m moshi.server --host 127.0.0.1 --port 8998"
if [ -n "$VOICES_DIR" ]; then
    WORKER_CMD="$WORKER_CMD --voice-prompt-dir $VOICES_DIR"
fi

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
