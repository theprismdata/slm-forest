#!/bin/bash

# RunPod Training Script for Gemma3 12B Tax Law Q&A Fine-tuning
# This script should be run on RunPod with GPU support (A100 or H100 recommended)

set -e

echo "=== Gemma3 12B Tax Law Q&A Fine-tuning on RunPod ==="
echo "Starting training setup..."

# Check if we're on RunPod
if [ -z "$RUNPOD_POD_ID" ]; then
    echo "Warning: Not running on RunPod environment"
else
    echo "Running on RunPod pod: $RUNPOD_POD_ID"
fi

# Check GPU availability and memory
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    
    # Check if we have enough GPU memory for 12B model
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "Available GPU memory: ${GPU_MEMORY}MB"
    
    if [ "$GPU_MEMORY" -lt 40000 ]; then
        echo "Warning: GPU memory is less than 40GB. Consider using smaller batch size or more aggressive quantization."
    fi
else
    echo "Warning: nvidia-smi not found. GPU may not be available."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r ../requirements-finetunning.txt

# Additional packages for Gemma3 12B
pip install accelerate>=0.20.0
pip install bitsandbytes>=0.41.0
pip install peft>=0.4.0
pip install transformers>=4.30.0
pip install torch>=2.0.0
pip install datasets>=2.12.0

# Check if dataset exists
if [ ! -f "../slm_datagen/fine-tunning-ds/distillation_legal_qa_dataset.json" ]; then
    echo "Error: Dataset file not found!"
    echo "Please ensure the dataset file exists at: ../slm_datagen/fine-tunning-ds/distillation_legal_qa_dataset.json"
    exit 1
fi

# Check dataset size
DATASET_SIZE=$(wc -l < ../slm_datagen/fine-tunning-ds/distillation_legal_qa_dataset.json)
echo "Dataset size: $DATASET_SIZE lines"

# Create output directory
mkdir -p gemma3-12b-tax-law-finetuned

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE="/tmp/transformers_cache"
export HF_DATASETS_CACHE="/tmp/datasets_cache"
export WANDB_DISABLED=true
export WANDB_MODE=disabled

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1

# For multi-GPU setup (if available)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Check available disk space
echo "Checking disk space..."
df -h

# Start training with error handling
echo "Starting Gemma3 12B training..."
python finetuning-gemma3.py 2>&1 | tee training.log

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    
    # Show final model size
    if [ -d "gemma3-12b-tax-law-finetuned" ]; then
        echo "Final model directory size:"
        du -sh gemma3-12b-tax-law-finetuned/
    fi
    
    # Show training log summary
    echo "Training log summary:"
    tail -20 training.log
    
else
    echo "Training failed. Check training.log for details."
    echo "Last 50 lines of training log:"
    tail -50 training.log
    exit 1
fi

echo "=== Training Complete ===" 