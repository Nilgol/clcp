
# CLCL - Camera-Lidar Contrastive Learning

This repository accompanies the master's thesis *Domain Generalization for Vision Perception Models by Camera-Lidar Contrastive Learning* (by Nils Golombiewski, TU Berlin, August 2024) and contains all relevant code used for experiments.

The thesis, including an abstract, as well as some presentation slides are included in the repository.

The source code is split into two distinct pipelines:
- **Pretraining Pipeline**: Developed using PyTorch Lightning. This pipeline is used for multimodal pretraining by aligning an image encoder with a point cloud encoder via a contrastive loss. 
- **Fine-tuning and Evaluation Pipeline**: Utilizes the MMSegmentation/MMPretrain framework from OpenMMLab. This pipeline is used to finetune the pretrained image encoder for semantic segmentation, then evaluating performace under distribution shift.

## Project Structure

- `pretrain/`: Contains pretraining pipeline source code.
- `finetune_eval/`: Contains source code alterations and configuration files for MMSegmentation/MMPretrain framework.
- `tools/`: General utility scripts for setup, data processing, and miscellaneous tasks.
- `envs/`: Snapshots of the mamba/conda environments used for development.

## Installation and Setup

During development, we used two separate environments, one for pretraining, one for finetuning and eval, that are documented under `envs/`. Unfortunately, directly recreating these with mamba does not work for us, so we recommend manual installation with the following procedure:

- Pretraining pipeline:

Clone `´pretrain/` and recreate the pretrain environment.
```
mamba create -n clcl_pretrain python=3.8
mamba activate clcl_pretrain
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
mamba install ninja pytorch-scatter ftfy regex timm
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'
mim install "mmdet3d>=1.1.0"
```
After that, install all other required dependencies (specifically `pytorch-lightning==2.3.1`) with pip.

- Finetuning/evaluation pipeline:

The above base environment also works for the finetuning/eval pipeline, but in order to include our source code alterations, it is necessary to install MMSegmentation and MMPretrain from source as per their instructions (in editable mode), then copy the contents from `finetune_eval/mmpretrain` and `finetune_eval/mmsegmentation` into the respective local directory. 

## Usage

- 