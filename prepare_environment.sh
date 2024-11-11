#!/bin/bash
!pip install opencv-python matplotlib
!pip install 'git+https://github.com/facebookresearch/segment-anything-2.git'

!mkdir -p images
!wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/notebooks/images/cars.jpg

!mkdir -p ../checkpoints/
!wget -P ../checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# INSTALL CLIP
!pip install ftfy regex tqdm
!pip install git+https://github.com/openai/CLIP.git

# INSTALL GOOGLE GENAI
!pip install -U -q google-generativeai