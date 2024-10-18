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

############### LEGACY CODE, not maintained ####################
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
####################################################################################


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


def load_model(cfg: Config) -> CameraLidarPretrain:
    """Load a model based on the provided configuration. Load from checkpoint if specified.

    WARNING: Checkpoint loading is untested.

    Args:
        cfg (Config): Configuration object containing model parameters.

    Returns:
        LightningModule: The model (either newly initialized or loaded from a checkpoint).
    """
    if cfg.get("checkpoint_path") and cfg.get("load_only_model"):
        return CameraLidarPretrain.load_from_checkpoint(checkpoint_path=cfg.get("checkpoint_path"))
    return CameraLidarPretrain(
        embed_dim=cfg.get("embed_dim", 384),
        temperature=cfg.get("temperature", 0.07),
        batch_size=cfg.get("batch_size", 32),
        learning_rate=cfg.get("learning_rate", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-5),
        freeze_lidar_encoder=cfg.get("freeze_lidar_encoder", False),
        projection_type=cfg.get("projection_type", "linear"),
    )


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


def _get_optimizer_params(cfg: Config) -> dict:
    """Retrieve parameters for AdamW optimizer from the configuration or set defaults.
    """
    optimizer_params = cfg.get("optimizer_params", {})

    return {
        "learning_rate": optimizer_params.get("learning_rate", 1e-4),
        "weight_decay": optimizer_params.get("weight_decay", 1e-5),
        "betas": optimizer_params.get("betas", (0.9, 0.98)),
    }


def _get_scheduler_params(cfg: Config) -> dict:
    """Retrieve parameters for cosine scheduler with warmup from the configuration or set defaults.
    """
    scheduler_params = cfg.get("scheduler_params", {})
    warmup_params = scheduler_params.get("warmup", {})
    cosine_params = scheduler_params.get("cosine", {})

    return {
        "warmup": {
            "start_factor": warmup_params.get("start_factor", 1e-2),
            "total_iters": warmup_params.get("total_iters", 5),
        },
        "cosine": {
            "T_0": cosine_params.get("T_0", 1),  # Period of the first restart
            "T_mult": cosine_params.get("T_mult", 2),  # Factor to increase period after each restart
            "eta_min": cosine_params.get("eta_min", 1e-5),  # Minimum learning rate
        }
    }


def _get_accelerator() -> str:
    """Determine and return the type of accelerator ('gpu' or 'cpu').
    """
    return "gpu" if torch.cuda.is_available() else "cpu"


def _get_number_of_gpus() -> int:
    """Determine and return the number of available GPUs.
    """
    available_gpus = torch.cuda.device_count()
    return available_gpus or 1


def _get_strategy(num_gpus: int) -> Union[DDPStrategy, SingleDeviceStrategy]:
    """Determine and return the distributed training strategy based on the number of GPUs.
    """
    if num_gpus > 1:
        return DDPStrategy(find_unused_parameters=True)
    return SingleDeviceStrategy(device=0)


def _get_callbacks(cfg: Config) -> List:
    """Set up callbacks for the trainer.
    """
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
    """Set up the WandB logger for experiment tracking.
    """
    log_dir = cfg.get("log_dir", LOG_DIR)
    return WandbLogger(save_dir=log_dir, name=cfg.exp_name)
