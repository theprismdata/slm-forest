#!/bin/bash

# RunPod Training Script for Phi-2 Tax Law Q&A Fine-tuning
# This script should be run on RunPod with GPU support

set -e

echo "=== Phi-2 Tax Law Q&A Fine-tuning on RunPod ==="
echo "Starting training setup..."

# Check if we're on RunPod
if [ -z "$RUNPOD_POD_ID" ]; then
    echo "Warning: Not running on RunPod environment"
else
    echo "Running on RunPod pod: $RUNPOD_POD_ID"
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "Warning: nvidia-smi not found. GPU may not be available."
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements-train.txt

# Check if dataset exists
if [ ! -f "fine-tunning-ds/distillation_legal_qa_dataset.json" ]; then
    echo "Error: Dataset file not found!"
    echo "Please ensure the dataset file exists at: fine-tunning-ds/distillation_legal_qa_dataset.json"
    exit 1
fi

# Check dataset size
DATASET_SIZE=$(wc -l < fine-tunning-ds/distillation_legal_qa_dataset.json)
echo "Dataset size: $DATASET_SIZE lines"

# Create output directory
mkdir -p phi2-tax-law-finetuned

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE="/tmp/transformers_cache"
export WANDB_DISABLED=true
export WANDB_MODE=disabled

# Start training
echo "Starting training..."
python train.py 2>&1 | tee training.log

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!" | tee -a training.log
    
    # Show model size
    if [ -d "phi2-tax-law-finetuned" ]; then
        echo "Model files:"
        ls -la phi2-tax-law-finetuned/
        
        # Calculate total size
        MODEL_SIZE=$(du -sh phi2-tax-law-finetuned/ | cut -f1)
        echo "Total model size: $MODEL_SIZE"
    fi
    
    echo "Model is ready for download and use on M4 MacBook Pro!"
else
    echo "Training failed!"
    exit 1
fi 