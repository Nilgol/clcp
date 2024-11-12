# CLCL - Camera-Lidar Contrastive Learning

This repository contains two main pipelines designed for multimodal contrastive learning to improve vision perception models, particularly for domain generalization in autonomous driving. The project uses image and point cloud data in both pretraining and fine-tuning stages, with distinct frameworks optimized for each phase:

- **Pretraining Pipeline**: Developed using PyTorch Lightning.
- **Fine-tuning and Evaluation Pipeline**: Utilizes the MMSegmentation framework from OpenMMLab.

## Key Features
- Self-supervised contrastive learning between camera and lidar data.
- Modular code for training, fine-tuning, and evaluation.
- Supports custom configurations and testing on ACDC datasets.

## Project Structure
- `pretrain/`: Contains pretraining pipeline source code.
- `finetune_eval/`: Code for fine-tuning and evaluating models on target datasets.
- `configs/`: Sample configuration files for each stage.
- `tests/`: Test files for validation and code correctness.
- `tools/`: General utility scripts for setup, data processing, and miscellaneous tasks.

## Requirements
- Python >= 3.8
- CUDA support (recommended for large-scale model training)
- Detailed dependencies are listed in `pretrain_env.yaml` and `finetune_eval_env.yaml`.

## Getting Started
See [INSTALL.md](./INSTALL.md) for setup instructions and [USAGE.md](./USAGE.md) for usage details.
