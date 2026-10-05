#!/usr/bin/env bash
# ==============================================================================
# PersonaPlex Multi-Worker Process Cluster Manager (Linux GPU)
# Runs N independent PersonaPlex workers bound strictly to localhost:8998+
# Traps exit signals to guarantee zero zombie GPU memory.
# ==============================================================================

set -euo pipefail

PIDS=()
WORKER_PORTS=()

cleanup() {
    echo ""
    echo "[CLUSTER] Received shutdown signal. Terminating all worker subprocesses..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "[CLUSTER] Stopping process PID $pid..."
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    sleep 1
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    echo "[CLUSTER] Clean shutdown completed. GPU memory released."
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# 1. GPU VRAM Assessment
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 || echo "0")
echo "=============================================================================="
echo "  PersonaPlex Multi-Worker Cluster Manager"
echo "  Detected GPU VRAM: ${VRAM_MB} MB"
echo "=============================================================================="

# Determine recommended concurrent worker processes based on VRAM:
# Each 7B full-precision worker consumes ~18,000 MB VRAM
if [ "$VRAM_MB" -ge 70000 ]; then
    NUM_WORKERS=3
    echo "[CAPACITY] 80GB GPU detected: Starting 3 concurrent worker streams."
elif [ "$VRAM_MB" -ge 40000 ]; then
    NUM_WORKERS=2
    echo "[CAPACITY] 48GB GPU detected: Starting 2 concurrent worker streams."
else
    NUM_WORKERS=1
    echo "[CAPACITY] 24GB GPU detected: Starting 1 active worker stream (recommended limit)."
fi

# Allow override via environment variable
NUM_WORKERS=${CONCURRENT_WORKERS:-$NUM_WORKERS}
START_PORT=${START_PORT:-8998}

echo "[CLUSTER] Launching $NUM_WORKERS worker processes..."
WORKER_ADDRS=()

for ((i=0; i<NUM_WORKERS; i++)); do
    PORT=$((START_PORT + i))
    WORKER_PORTS+=("$PORT")
    WORKER_ADDRS+=("127.0.0.1:$PORT")
    echo "[CLUSTER] Starting PersonaPlex worker on 127.0.0.1:$PORT..."
    python3 -m moshi.server \
        --host 127.0.0.1 \
        --port "$PORT" \
        --voice-prompt-dir "${HF_HOME:-$HOME/.cache/huggingface}/voices" &
    PIDS+=("$!")
done

# Format comma-separated worker addresses for gateway
WORKER_ARGS=$(IFS=,; echo "${WORKER_ADDRS[*]}")
echo "[CLUSTER] Workers running: $WORKER_ARGS"
echo "[CLUSTER] Launching Orchestration Gateway on 0.0.0.0:8000..."

python3 -m orchestration.cli run-gateway \
    --host 0.0.0.0 \
    --port 8000 \
    --worker "$WORKER_ARGS" &
PIDS+=("$!")

# Wait for any process to exit
wait -n "${PIDS[@]}"
cleanup
