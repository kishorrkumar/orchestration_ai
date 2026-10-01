#!/usr/bin/env bash
# ==============================================================================
# Hardened PersonaPlex & Orchestration Setup for Krutrim Cloud (Ubuntu GPU Instance)
# Targets: NVIDIA A10G / A100 / H100 / RTX 3090/4090 / L40 / Blackwell (Ubuntu 22.04+)
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo "=============================================================================="
echo "  PersonaPlex 7B Orchestration Layer: Krutrim Cloud GPU Setup"
echo "=============================================================================="

# --- 1. Hardware & Driver Preflight Check ---
log "1. Verifying NVIDIA GPU driver and hardware..."
if ! command -v nvidia-smi &> /dev/null; then
    error "nvidia-smi not found! Please attach an NVIDIA GPU and install NVIDIA drivers."
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
log "Detected GPU: ${GREEN}${GPU_NAME}${NC} with ${GREEN}${GPU_VRAM}${NC} (Driver: ${DRIVER_VERSION})"

# --- 2. System Audio & Codec Dependencies ---
log "2. Installing required system packages (libopus, ffmpeg, build-essential)..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git \
    git-lfs \
    build-essential \
    libopus-dev \
    libopus0 \
    ffmpeg \
    python3-pip \
    python3-venv \
    python3-dev \
    curl \
    pkg-config

# Verify ffmpeg and libopus
ffmpeg -version | head -n 1 || error "FFmpeg verification failed."
success "System packages verified."

# --- 3. Persistent Disk & Hugging Face Storage ---
log "3. Configuring persistent model storage..."
if [ -d "/workspace" ] && [ -w "/workspace" ]; then
    export HF_HOME="/workspace/huggingface"
elif [ -d "/data" ] && [ -w "/data" ]; then
    export HF_HOME="/data/huggingface"
else
    export HF_HOME="$HOME/.cache/huggingface"
fi
mkdir -p "$HF_HOME"
log "HF_HOME set to: ${HF_HOME}"

# --- 4. Python Virtual Environment ---
log "4. Initializing Python virtual environment..."
VENV_PATH="$HOME/personaplex_env"
if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"
pip install --upgrade pip setuptools wheel

# --- 5. PyTorch with CUDA Matching Build ---
log "5. Installing CUDA-optimized PyTorch..."
# Detect CUDA version or default to cu124
CUDA_MAJOR=$(nvidia-smi | grep -o "CUDA Version: [0-9]*" | awk '{print $3}' | cut -d'.' -f1 || echo "12")
if [ "$CUDA_MAJOR" -ge "13" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

# Validate PyTorch CUDA support
python3 -c "
import torch
assert torch.cuda.is_available(), 'PyTorch CUDA is not available!'
print(f'PyTorch {torch.__version__} successfully initialized with CUDA device: {torch.cuda.get_device_name(0)}')
" || error "PyTorch CUDA initialization failed!"
success "PyTorch CUDA initialized successfully."

# --- 6. Hugging Face Authentication & Model License Verification ---
log "6. Verifying Hugging Face Token & PersonaPlex 7B License..."
if [ -z "${HF_TOKEN:-}" ]; then
    warn "HF_TOKEN environment variable is not set!"
    echo "Please accept the model license at https://huggingface.co/nvidia/personaplex-7b-v1"
    read -rp "Enter your Hugging Face User Access Token (with Read permission): " USER_HF_TOKEN
    export HF_TOKEN="$USER_HF_TOKEN"
fi

pip install huggingface_hub[cli]

python3 -c "
import os
from huggingface_hub import HfApi
api = HfApi()
token = os.environ.get('HF_TOKEN')
try:
    info = api.model_info('nvidia/personaplex-7b-v1', token=token)
    print(f'Hugging Face License verified! Model ID: {info.id}')
except Exception as e:
    raise RuntimeError(f'HF Model Access Error: {e}. Ensure you accepted the license at https://huggingface.co/nvidia/personaplex-7b-v1')
" || error "HF_TOKEN validation failed! Please check token permissions and accepted license."
success "Hugging Face authentication verified."

# --- 7. Upstream PersonaPlex (Moshi) Installation ---
log "7. Installing upstream PersonaPlex engine..."
PERSONAPLEX_DIR="$HOME/personaplex"
if [ ! -d "$PERSONAPLEX_DIR" ]; then
    git clone https://github.com/NVIDIA/personaplex.git "$PERSONAPLEX_DIR"
else
    git -C "$PERSONAPLEX_DIR" pull || true
fi
# Patch moshi pyproject.toml to relax <2.5 constraint for modern PyTorch / Python 3.13
if [ -f "$PERSONAPLEX_DIR/moshi/pyproject.toml" ]; then
    sed -i -E 's/torch<2.5,>=2.2.0/torch>=2.2.0/g; s/<2.5[0-9.]*//g' "$PERSONAPLEX_DIR/moshi/pyproject.toml" || true
fi
pip install rustymimi sounddevice einops sentencepiece || true
pip install --no-deps -e "$PERSONAPLEX_DIR/moshi"
pip install accelerate  # Required for cpu-offload support

# --- 8. Pre-downloading Model Weights & Voice Presets ---
log "8. Pre-downloading PersonaPlex 7B weights and 18 voice presets..."
python3 -c "
import os, tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download

repo = 'nvidia/personaplex-7b-v1'
token = os.environ.get('HF_TOKEN')

print('Downloading Mimi neural codec...')
hf_hub_download(repo, 'tokenizer-e351c8d8-checkpoint125.safetensors', token=token)

print('Downloading 32k text tokenizer...')
hf_hub_download(repo, 'tokenizer_spm_32k_3.model', token=token)

print('Downloading Moshi LM 7B checkpoint (14.5 GB)...')
hf_hub_download(repo, 'model.safetensors', token=token)

print('Downloading voice presets archive...')
voices_tgz = hf_hub_download(repo, 'voices.tgz', token=token)
voices_dir = Path(voices_tgz).parent / 'voices'
if not voices_dir.exists():
    print(f'Extracting {voices_tgz}...')
    with tarfile.open(voices_tgz, 'r:gz') as tar:
        tar.extractall(path=voices_tgz.parent)
print('Voice presets extracted to:', voices_dir)
"
success "All model weights and 18 voice presets downloaded to persistent cache."

# --- 9. Orchestration AI Layer Setup ---
log "9. Installing Orchestration AI Layer..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH_DIR="${SCRIPT_DIR}"
if [ ! -f "$ORCH_DIR/setup.py" ] && [ ! -f "$ORCH_DIR/pyproject.toml" ]; then
    ORCH_DIR="$HOME/orchestration_ai"
    if [ ! -d "$ORCH_DIR" ]; then
        git clone https://github.com/kishorrkumar/orchestration_ai.git "$ORCH_DIR"
    fi
fi
pip install -r "$ORCH_DIR/requirements.txt"
pip install -e "$ORCH_DIR"

# Ensure storage directories exist
mkdir -p "$ORCH_DIR/data/cloned_voices"
mkdir -p "$ORCH_DIR/data/transcripts"

# --- 10. Systemd Service Units Installation ---
log "10. Configuring systemd services..."
if [ -d "/etc/systemd/system" ] && command -v systemctl &> /dev/null; then
    sudo cp "$ORCH_DIR/deploy/systemd/personaplex-gateway.service" /etc/systemd/system/ || true
    sudo cp "$ORCH_DIR/deploy/systemd/personaplex-worker@.service" /etc/systemd/system/ || true
    sudo systemctl daemon-reload 2>/dev/null || true
    success "Systemd services installed (personaplex-gateway, personaplex-worker@)."
fi

success "=============================================================================="
success " PersonaPlex Orchestration Layer successfully installed on Krutrim Cloud!"
success "=============================================================================="
echo ""
echo "SYSTEM READY:"
echo "1. Activate environment: source ~/personaplex_env/bin/activate"
echo "2. Run test suite:       pytest -v"
echo "3. Start Gateway:        python -m orchestration.cli run-gateway --host 127.0.0.1 --port 8000 --worker 127.0.0.1:8998"
echo "   (or via systemd:      sudo systemctl start personaplex-gateway)"
echo "4. Reverse Proxy / HTTPS: Configure Caddy or Nginx from deploy/proxy/ with SSL certificate."
echo "   IMPORTANT: Browsers block getUserMedia (microphone) on non-HTTPS origins off-localhost."
echo "   Use Caddy automatic HTTPS (deploy/proxy/Caddyfile) to access /console securely."
echo "=============================================================================="

