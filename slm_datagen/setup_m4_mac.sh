#!/bin/bash

# M4 MacBook Pro Setup Script for Phi-2 Tax Law Q&A Inference
# This script sets up the environment for running the fine-tuned model

set -e

echo "=== M4 MacBook Pro Setup for Phi-2 Tax Law Q&A ==="
echo "Setting up inference environment..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: This script is designed for macOS"
    exit 1
fi

# Check if we have M1/M2/M3/M4 chip
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo "Warning: Not running on Apple Silicon (M1/M2/M3/M4)"
    echo "Current architecture: $ARCH"
    echo "Performance may be suboptimal"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with MPS support
echo "Installing PyTorch with MPS support..."
pip install torch torchvision torchaudio

# Install other dependencies
echo "Installing inference dependencies..."
pip install -r requirements-inference.txt

# Check MPS availability
echo "Checking MPS (Metal Performance Shaders) availability..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    print('MPS device will be used for inference')
else:
    print('MPS not available, will use CPU')
"

# Check if model directory exists
if [ ! -d "phi2-tax-law-finetuned" ]; then
    echo "Warning: Fine-tuned model directory not found!"
    echo "Please ensure you have downloaded the model from RunPod"
    echo "Expected location: ./phi2-tax-law-finetuned/"
else
    echo "Model directory found: phi2-tax-law-finetuned/"
    echo "Model files:"
    ls -la phi2-tax-law-finetuned/
fi

echo ""
echo "=== Setup Complete ==="
echo "To run inference, use one of the following commands:"
echo ""
echo "1. Interactive mode:"
echo "   python inference.py --interactive"
echo ""
echo "2. Single question:"
echo "   python inference.py --question '부가가치세 신고는 어떻게 하나요?'"
echo ""
echo "3. Batch questions from file:"
echo "   python inference.py --questions_file questions.txt --output_file answers.json"
echo ""
echo "4. Test with sample question:"
echo "   python inference.py --question '연말정산 시 의료비 공제는 어떻게 받나요?'" 