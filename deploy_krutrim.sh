#!/bin/bash
# ==============================================================================
# PersonaPlex & Orchestration Setup Script for Krutrim Cloud (Ubuntu GPU Instance)
# ==============================================================================

set -e

echo "=== 1. Updating system and installing dependencies ==="
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential libopus-dev ffmpeg python3-pip python3-venv

echo "=== 2. Setting up Python virtual environment ==="
python3 -m venv ~/personaplex_env
source ~/personaplex_env/bin/activate
pip install --upgrade pip

echo "=== 3. Cloning NVIDIA PersonaPlex upstream engine ==="
if [ ! -d "$HOME/personaplex" ]; then
    git clone https://github.com/NVIDIA/personaplex.git ~/personaplex
fi

echo "=== 4. Installing Moshi engine & dependencies ==="
pip install ~/personaplex/moshi

echo "=== 5. Cloning Orchestration Layer ==="
if [ ! -d "$HOME/orchestration_ai" ]; then
    git clone https://github.com/kishorrkumar/orchestration_ai.git ~/orchestration_ai
fi

echo "=== 6. Installing Orchestration Layer dependencies ==="
pip install -r ~/orchestration_ai/requirements.txt

echo "=============================================================================="
echo " Setup Complete!"
echo "=============================================================================="
echo ""
echo "NEXT STEPS:"
echo "1. Export your Hugging Face token (make sure you accepted the license at"
echo "   https://huggingface.co/nvidia/personaplex-7b-v1):"
echo "   export HF_TOKEN=\"hf_your_actual_token\""
echo ""
echo "2. Start the PersonaPlex 7B inference worker:"
echo "   python -m moshi.server --host 0.0.0.0 --port 8998"
echo ""
echo "3. In another terminal, start the Orchestration Gateway:"
echo "   python -m orchestration.cli run-gateway --host 0.0.0.0 --port 8000 --worker gpu-krutrim:127.0.0.1:8998"
echo ""
echo "4. Open http://<YOUR_VM_PUBLIC_IP>:8000/console in your browser to talk!"
echo "=============================================================================="
