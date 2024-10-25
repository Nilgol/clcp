"""A Module for multimodal self-supervised pretraining of an image and a point cloud encoder.
The module loads training and validation datasets, initializes a model, and trains it using PyTorch Lightning.
"""

import os
from typing import Tuple, List, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy, SingleDeviceStrategy
from pytorch_lightning import Trainer

from config import Config
from data.a2d2_dataset import A2D2Dataset
from data.a2d2_loader import build_loader
from image_point_cloud_pretrain import ImagePointCloudPretrain
from model.minkunet_encoder import MinkUNetEncoder
from model.vit_image_encoder import VitImageEncoder

# Some data paths. Might be moved to config for more flexibility.
DATA_ROOT_DIR = "/homes/math/golombiewski/workspace/data/A2D2"
WORK_DIR = "/homes/math/golombiewski/workspace/fast/clcl"
LOG_DIR = WORK_DIR + "/logs"
CHECKPOINT_SAVE_DIR = WORK_DIR + "/checkpoints"

def train(cfg: Config) -> None:
    """Train a model using the provided configuration:
    1. Prepare train and val dataloader.
    2. Load a model from config of previous checkpoint.
    3. Instantiate PyTorch Lightning Trainer, then call the `fit` method on it.

    Args:
        cfg (Config): The configuration object containing all parameters.
    """
    train_loader, val_loader = get_dataloaders(cfg)
    model = load_model(cfg)
    trainer = load_trainer(cfg)

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.get("checkpoint_path") if not cfg.get("load_only_model") else None,
    )


def get_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """Prepare and return training and validation dataloaders using the given configuration.

    Args:
        cfg (Config): The configuration containing the parameters.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
            Training and validation dataloaders.
    """
    train_dataset = A2D2Dataset(
        root_path=DATA_ROOT_DIR,
        val_ratio=cfg.get("val_ratio", 0.15),
        split="train",
        augment=cfg.get("augment", False),
    )
    val_dataset = A2D2Dataset(
        root_path=DATA_ROOT_DIR,
        val_ratio=cfg.get("val_ratio", 0.15),
        split="val",
        augment=False,
    )
    train_loader = build_loader(
        train_dataset,
        batch_size=cfg.get("batch_size", 32),
        num_workers=cfg.get("num_workers", 8),
        shuffle=True,
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=cfg.get("batch_size", 32),
        num_workers=cfg.get("num_workers", 8),
        shuffle=False,
    )
    return train_loader, val_loader


def load_model(cfg: Config) -> ImagePointCloudPretrain:
    """Load a model based on the provided configuration. Load from checkpoint if specified.

    Currently supported encoder models: ViT for images, MinkUNet for point clouds.
    Add new models here.
    
    WARNING: Checkpoint loading is untested.

    Args:
        cfg (Config): Configuration object containing model parameters.

    Returns:
        LightningModule: The training model.
    """
    # Load model from checkpoint if specified
    if cfg.get("checkpoint_path") and cfg.get("load_only_model"):
        return ImagePointCloudPretrain.load_from_checkpoint(
            checkpoint_path=cfg.get("checkpoint_path")
        )

    # Set image encoder, add new encoder types here
    image_encoder_type = cfg.get("image_encoder_type", "vit")
    if image_encoder_type == "vit":
        image_encoder = VitImageEncoder(embed_dim=cfg.get("embed_dim", 384))
    else:
        raise ValueError(f"Unsupported image encoder type: {image_encoder_type}")

    # Set point cloud encoder, add new encoder types here
    point_cloud_encoder_type = cfg.get("point_cloud_encoder_type", "minkunet")
    if point_cloud_encoder_type == "minkunet":
        point_cloud_encoder = MinkUNetEncoder(**_get_point_cloud_encoder_params(cfg))
    else:
        raise ValueError(f"Unsupported point cloud encoder type: {point_cloud_encoder_type}")

    image_point_cloud_pretrain = ImagePointCloudPretrain(
        image_encoder=image_encoder,
        point_cloud_encoder=point_cloud_encoder,
        temperature=cfg.get("temperature", 0.07),
        batch_size=cfg.get("batch_size", 32),
        optimizer_params=_get_optimizer_params(cfg),
        scheduler_params=_get_scheduler_params(cfg),
    )

    return image_point_cloud_pretrain


def load_trainer(cfg: Config) -> Trainer:
    """Initialize and return a PyTorch Lightning Trainer instance with the specified configuration.

    Args:
        cfg (Config): Configuration object containing trainer parameters.

    Returns:
        Trainer: A PyTorch Lightning Trainer object.
    """
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    accelerator = _get_accelerator()
    num_gpus = _get_number_of_gpus()
    strategy = _get_strategy(num_gpus)
    callbacks = _get_callbacks(cfg)
    logger = _get_logger(cfg)

    trainer = Trainer(
        fast_dev_run=2,
        logger=logger,
        precision="32-true",
        accelerator=accelerator,
        devices=num_gpus,
        limit_train_batches=None,
        max_epochs=cfg.get("max_epochs", 100),
        strategy=strategy,
        callbacks=callbacks,
    )

    return trainer

def _get_point_cloud_encoder_params(cfg: Config) -> dict:
    """Retrieve parameters for the point cloud encoder from configuration or set defaults."""
    encoder_params = cfg.get("point_cloud_encoder_params", {})

    return {
        "embed_dim": encoder_params.get("embed_dim", 384),
        "freeze_encoder_weights": encoder_params.get("freeze_encoder_weights", False),
        "projection_type": encoder_params.get("projection_type", "linear"),
    }

def _get_optimizer_params(cfg: Config) -> dict:
    """Retrieve parameters for AdamW optimizer from the configuration or set defaults."""
    optimizer_params = cfg.get("optimizer", {})

    return {
        "learning_rate": optimizer_params.get("learning_rate", 1e-4),
        "weight_decay": optimizer_params.get("weight_decay", 1e-5),
        "betas": optimizer_params.get("betas", (0.9, 0.98)),
    }


def _get_scheduler_params(cfg: Config) -> dict:
    """Retrieve parameters for cosine scheduler with warmup from the configuration or set defaults."""
    scheduler_params = cfg.get("scheduler", {})
    warmup_params = scheduler_params.get("warmup", {})
    cosine_params = scheduler_params.get("cosine", {})

    return {
        "warmup": {
            "start_factor": warmup_params.get("start_factor", 1e-2),
            "total_iters": warmup_params.get("total_iters", 5),
        },
        "cosine": {
            "T_0": cosine_params.get("T_0", 1),  # Period of the first restart
            "T_mult": cosine_params.get(
                "T_mult", 2
            ),  # Factor to increase period after each restart
            "eta_min": cosine_params.get("eta_min", 1e-5),  # Minimum learning rate
        },
    }


def _get_accelerator() -> str:
    """Determine and return the type of accelerator ('gpu' or 'cpu')."""
    return "gpu" if torch.cuda.is_available() else "cpu"


def _get_number_of_gpus() -> int:
    """Determine and return the number of available GPUs."""
    available_gpus = torch.cuda.device_count()
    return available_gpus or 1


def _get_strategy(num_gpus: int) -> Union[DDPStrategy, SingleDeviceStrategy]:
    """Determine and return the distributed training strategy based on the number of GPUs."""
    if num_gpus > 1:
        return DDPStrategy(find_unused_parameters=True)
    return SingleDeviceStrategy(device=0)


def _get_callbacks(cfg: Config) -> List:
    """Set up callbacks for the trainer."""
    exp_name = cfg.exp_name
    checkpoint_save_dir = cfg.get("checkpoint_save_dir", CHECKPOINT_SAVE_DIR)
    experiment_checkpoint_dir = os.path.join(checkpoint_save_dir, exp_name)
    os.makedirs(experiment_checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_checkpoint_dir,
        filename=exp_name + "_{epoch:02d}_{val_loss:.2f}",
        save_top_k=-1,
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_on_train_epoch_end=True,
        verbose=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=50, mode="min", verbose=True
    )
    learning_rate_callback = LearningRateMonitor(logging_interval="step")

    return [checkpoint_callback, learning_rate_callback, early_stopping_callback]


def _get_logger(cfg: Config) -> WandbLogger:
    """Set up the WandB logger for experiment tracking."""
    log_dir = cfg.get("log_dir", LOG_DIR)
    return WandbLogger(save_dir=log_dir, name=cfg.exp_name)
