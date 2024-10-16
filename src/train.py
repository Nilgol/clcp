"""Training script for the Camera-Lidar Contrastive Learning (CLCL) model.
"""
import os
from typing import Tuple

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
from pytorch_lightning import LightningModule, Trainer

from config import Config
from data.a2d2_dataset import A2D2Dataset
from data.a2d2_loader import build_loader
from model.lidar_encoder_minkunet import LidarEncoderMinkUNet
from model.image_encoder import ImageEncoder

# Path for A2D2 dataset. Should usually not change, so we set it here once.
DATA_ROOT_DIR = "/homes/math/golombiewski/workspace/data/A2D2"
WORK_DIR = "/homes/math/golombiewski/workspace/fast/clcl"
LOG_DIR = WORK_DIR + "/logs"
CHECKPOINT_SAVE_DIR = WORK_DIR + "/checkpoints"


class CameraLidarPretrain(LightningModule):
    """
    A module for contrastive pretraining of pairs of camera images and lidar point clouds.

    The model takes embeddings from an image and point cloud encoder, respectively, and computes
    as contrastive loss between them.

    Attributes:
        embed_dim (int): Dimension of joint embedding space for image and point cloud encoders.
        temperature (float): Temperature parameter for the contrastive loss to scale logits.
        learning_rate (float): Learning rate for the AdamW optimizer.
        weight_decay (float): Weight decay for the AdamW optimizer.
        batch_size (int): The batch size for training.
        epoch_size (float): The number of batches per epoch (determined dynamically).
        freeze_lidar_encoder (bool): If True, freezes the point cloud encoder weights.
        projection_type (str): The type of projection head, either "linear" or "mlp".

    Methods:
        contrastive_loss(image_embeddings, lidar_embeddings):
            Computes the contrastive loss between normalized image and point cloud embeddings.
        
        training_step(batch, batch_idx):
            Performs a forward pass and computes the training loss for a given batch.
        
        validation_step(batch, batch_idx):
            Performs a forward pass and computes the validation loss for a given batch.
        
        configure_optimizers():
            Configures the AdamW optimizer and learning rate schedulers.
    """
    def __init__(
        self,
        embed_dim,
        temperature,
        learning_rate,
        weight_decay,
        batch_size,
        epoch_size,
        freeze_lidar_encoder=False,
        projection_type="linear",
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.lidar_encoder = LidarEncoderMinkUNet(
            embed_dim=embed_dim,
            freeze_encoder=freeze_lidar_encoder,
            projection_type=projection_type,
        )
        self.embed_dim = embed_dim
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def contrastive_loss(self, image_embeddings, lidar_embeddings):
        """Normalize image and point cloud embeddings, then compute the contrastive loss.
        """
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        lidar_embeddings = F.normalize(lidar_embeddings, p=2, dim=-1)
        logits = (image_embeddings @ lidar_embeddings.T) / torch.exp(self.temperature)
        labels = torch.arange(image_embeddings.size(0), device=self.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_l = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_l) / 2

    def _common_step(self, batch):
        images, point_clouds = batch
        image_embeddings = self.image_encoder(images)
        lidar_embeddings = self.lidar_encoder(point_clouds, self.batch_size)
        loss = self.contrastive_loss(image_embeddings, lidar_embeddings)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "learning_rate",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log("temperature", self.temperature, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            # betas=(0.9, 0.98),
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6 / self.learning_rate, total_iters=5
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=1,  # Number of iterations for the first restart
            T_mult=2,  # A factor increases T_i after a restart
            eta_min=1e-5,  # Minimum learning rate
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[5],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


def prepare_data_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """Prepare training and validation data loaders using the given configuration.

    Args:
        cfg (Config): The configuration object containing the parameters.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
            Training and validation data loaders.
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


def load_model(cfg: Config, train_loader: torch.utils.data.DataLoader) -> CameraLidarPretrain:
    """Load a model based on the provided configuration. Load from checkpoint if specified.

    Args:
        cfg (Config): Configuration object containing model parameters.
        train_loader (torch.utils.data.DataLoader): The training data loader (for epoch size).
        devices (int): Number of devices used for distributed training.

    Returns:
        LightningModule: The model (either newly initialized or loaded from a checkpoint).
    """
    available_gpus = torch.cuda.device_count() or None
    devices = available_gpus if available_gpus else 1

    if cfg.get("checkpoint_path") and cfg.get("load_only_model"):
        return CameraLidarPretrain.load_from_checkpoint(checkpoint_path=cfg.get("checkpoint_path"))
    return CameraLidarPretrain(
        embed_dim=cfg.get("embed_dim", 384),
        temperature=cfg.get("temperature", 0.07),
        learning_rate=cfg.get("learning_rate", 1e-4),
        batch_size=cfg.get("batch_size", 32),
        freeze_lidar_encoder=cfg.get("freeze_lidar_encoder", False),
        epoch_size=len(train_loader) / devices,
        projection_type=cfg.get("projection_type", "linear"),
        weight_decay=cfg.get("weight_decay", 1e-5),
    )


def load_trainer(cfg: Config) -> Trainer:
    """Initialize and return a PyTorch Lightning Trainer with the specified configuration.

    Args:
        cfg (Config): Configuration object containing trainer parameters.

    Returns:
        Trainer: A PyTorch Lightning Trainer object.
    """
    exp_name = cfg.exp_name
    available_gpus = torch.cuda.device_count() or None
    accelerator = "gpu" if available_gpus else "cpu"
    devices = available_gpus if available_gpus else 1

    # Set up accelerator and strategy
    available_gpus = torch.cuda.device_count() or None
    accelerator = "gpu" if available_gpus else "cpu"
    devices = available_gpus if available_gpus else 1
    strategy = (
        DDPStrategy(find_unused_parameters=True)
        if devices > 1
        else SingleDeviceStrategy(device=0)
    )
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # Set up callbacks
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
        # every_n_train_steps=250,
        save_on_train_epoch_end=True,
        verbose=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=50, mode="min", verbose=True
    )
    learning_rate_callback = LearningRateMonitor(logging_interval="step")

    # Set up logger
    log_dir = cfg.get("log_dir", LOG_DIR)
    wandb_logger = WandbLogger(save_dir=log_dir, name=exp_name)

    # Initialize trainer
    trainer = Trainer(
        # detect_anomaly=True,
        fast_dev_run=2,
        logger=wandb_logger,
        precision="32-true",
        accelerator=accelerator,
        devices=devices,
        limit_train_batches=None,
        max_epochs=cfg.get("max_epochs", 100),
        strategy=strategy,
        callbacks=[checkpoint_callback, learning_rate_callback, early_stopping_callback],
    )

    return trainer


def train(cfg: Config) -> None:
    """Train a model using the provided configuration.

    Args:
        cfg (Config): The configuration object containing all parameters.
    """
    train_loader, val_loader = prepare_data_loaders(cfg)
    model = load_model(cfg, train_loader)
    trainer = load_trainer(cfg)

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.get("checkpoint_path") if not cfg.get("load_only_model") else None,
    )
