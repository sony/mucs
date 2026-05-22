#!/bin/bash

# We assume python>=3.10

# Create environment
python3 -m venv .venv
source .venv/bin/activate

# General utilities
pip install icdiff==2.0.7

# Pytorch basics
pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install lightning==2.5.1 tensorboard==2.19.0 einops==0.8.1
pip install pandas

# For eval script (LPIPS; also used as attribution method)
pip install lpips
pip install scikit-learn==1.7.2

# For captions (CLIP; also used as attribution method)
pip install open_clip_torch==2.32.0

# For FID calculation
pip install torch-fidelity==0.3.0

# Pytorch utilities
pip install torchinfo==1.8.0 omegaconf==2.3.0 matplotlib==3.10.3

# For preprocessing COCO
pip install pycocotools==2.0.10

# For downloading TinyAutoencoder weights
cd pointer_to/cache/
git clone https://github.com/madebyollin/taesd.git
cd ../..
