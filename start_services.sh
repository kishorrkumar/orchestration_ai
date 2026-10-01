#!/usr/bin/env bash
# ==============================================================================
# Launch PersonaPlex 7B Worker and Orchestration Gateway
# Accessible externally via Krutrim Cloud HTTP Service [Port 8000]
# ==============================================================================

set -euo pipefail

# 1. Activate Python virtual environment
if [ -d "$HOME/personaplex_env" ]; then
    source "$HOME/personaplex_env/bin/activate"
elif [ -d "/workspace/personaplex_env" ]; then
    source "/workspace/personaplex_env/bin/activate"
fi

# 2. Configure Persistent HF Cache
if [ -d "/workspace/huggingface" ]; then
    export HF_HOME="/workspace/huggingface"
elif [ -d "/data/huggingface" ]; then
    export HF_HOME="/data/huggingface"
fi

# 3. Locate Voice Presets
VOICES_DIR=""
if [ -d "$HF_HOME/voices" ]; then
    VOICES_DIR="$HF_HOME/voices"
else
    # Find extracted voices folder in HF hub cache
    FOUND=$(find "$HF_HOME" -type d -name "voices" 2>/dev/null | head -n 1)
    if [ -n "$FOUND" ]; then
        VOICES_DIR="$FOUND"
    fi
fi

echo "=============================================================================="
echo " Starting PersonaPlex 7B Real-Time Voice Services"
echo " HF_HOME:    ${HF_HOME:-$HOME/.cache/huggingface}"
echo " Voices Dir: ${VOICES_DIR:-none}"
echo "=============================================================================="

# 4. Check if Moshi Worker is already running on port 8998
if lsof -i :8998 >/dev/null 2>&1 || ss -lntp 2>/dev/null | grep -q ":8998"; then
    echo "[INFO] Worker already running on port 8998."
else
    echo "[INFO] Launching PersonaPlex 7B Moshi Worker on 127.0.0.1:8998 (GPU)..."
    WORKER_CMD="python -m moshi.server --host 127.0.0.1 --port 8998"
    if [ -n "$VOICES_DIR" ]; then
        WORKER_CMD="$WORKER_CMD --voice-prompt-dir $VOICES_DIR"
    fi
    $WORKER_CMD > worker.log 2>&1 &
    WORKER_PID=$!
    echo "[SUCCESS] Worker started in background (PID: $WORKER_PID, logs: worker.log)"
    echo "Waiting for worker to initialize model weights..."
    sleep 3
fi

# 5. Launch Orchestration Gateway on 0.0.0.0:8000
echo "[INFO] Launching Orchestration Gateway on 0.0.0.0:8000..."
echo "=============================================================================="
echo " ✨ CONSOLE READY: Access via Krutrim Cloud 'HTTP Service [Port 8000]'"
echo " Or open: http://<EXTERNAL_IP>:8000/console"
echo "=============================================================================="

python -m orchestration.cli run-gateway \
    --host 0.0.0.0 \
    --port 8000 \
    --worker 127.0.0.1:8998 \
    --worker-type personaplex
