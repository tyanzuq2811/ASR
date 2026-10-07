#!/usr/bin/env bash
# ==============================================================================
# Script setup & execution on Linux Server (NVIDIA A30 GPU / 24GB VRAM)
# Directory: /hdd3/users/dunglt/asr
# ==============================================================================
set -e

echo "=== [1/4] Checking NVIDIA GPU Environment ==="
nvidia-smi

echo "=== [2/4] Setting up Python Environment ==="
# Initialize Conda in bash script
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
fi

# Create environment if it doesn't exist
if ! conda info --envs | grep -q "asr_env"; then
    echo "Creating new conda environment 'asr_env' from environment.yml..."
    conda env create -f environment.yml
fi

echo "Activating 'asr_env'..."
conda activate asr_env

echo "Ensuring whisper-medusa package is registered..."
pip install -e ./whisper-medusa --no-deps

echo "=== [3/4] Running Comprehensive ASR Benchmark (All 200 Samples) ==="
python scripts/run_benchmark.py \
    --medusa-model "aiola/whisper-medusa-v1" \
    --baseline-model "openai/whisper-large-v2" \
    --output-dir "benchmark_results"

echo "=== [4/4] Completed! Results are saved in 'benchmark_results/' ==="
ls -lh benchmark_results/

echo ""
echo "💡 Để chạy Web Demo trên server (truy cập qua trình duyệt):"
echo "   python src/server.py"
echo "   (Sau đó mở SSH tunnel từ máy local: ssh -L 8000:localhost:8000 dunglt@aiotlab-ws1)"
