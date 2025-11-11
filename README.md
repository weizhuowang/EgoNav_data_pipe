# Data pipeline for the EgoNav project
This repository contains the code for the data pipeline of the EgoNav project. The data pipeline is responsible for processing the raw data collected by the Smartbelt system and preparing it for the training of the diffusion models.

This is designed to run on a slurm cluster with many convenient scripts and tools to fully utilize the performance.

## Installation

### Prerequisites
1. Install the egonav package following the instructions in the MultiModalTraj repository:

2. Clone the DINOv2 repository:
```bash
cd /home/weizhuo2/Documents/gits
git clone https://github.com/facebookresearch/dinov2.git
```

3. Setup prereq for MMCV (probably need to recompile it on newer 5090)
```bash
mamba install -c nvidia/label/cuda-12.8.0 cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX && export PATH=$CONDA_PREFIX/bin:$PATH && which nvcc && nvcc --version
```

### Additional Dependencies for DINOv2_labeler
```bash
# Install MMlab packages for semantic segmentation
pip install openmim
mim install mmcv-full==1.*
pip install mmsegmentation==0.30.0
# pip install ftfy
```

edit ~/miniforge3/envs/egonav/lib/python3.10/site-packages/mmseg/__init__.py to relax the version constraints

Update the `REPO_PATH` in `Scripts/DINOv2_labeler.py` (line 4) to point to your dinov2 clone location.