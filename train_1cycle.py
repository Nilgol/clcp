import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy, SingleDeviceStrategy

from data.a2d2_dataset import A2D2Dataset
from data.a2d2_loader import build_loader
from model.lidar_encoder_minkunet import LidarEncoderMinkUNet
from model.image_encoder import ImageEncoder


class CameraLidarPretrain(pl.LightningModule):
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
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        lidar_embeddings = F.normalize(lidar_embeddings, p=2, dim=-1)
        logits = (image_embeddings @ lidar_embeddings.T) / torch.exp(self.temperature)
        labels = torch.arange(image_embeddings.size(0), device=self.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_l = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_l) / 2

    def _common_step(self, batch, batch_idx):
        images, point_clouds = batch
        image_embeddings = self.image_encoder(images)
        lidar_embeddings = self.lidar_encoder(point_clouds, self.batch_size)
        loss = self.contrastive_loss(image_embeddings, lidar_embeddings)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
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
        loss = self._common_step(batch, batch_idx)
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
            betas=(0.9, 0.98),
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,  # Peak learning rate
            total_steps=1000,
            pct_start=0.5,  # Percentage of the cycle spent increasing LR
            anneal_strategy='cos',  # Cosine annealing
            div_factor=1e3,  # Initial learning rate is max_lr / div_factor
            final_div_factor=1e3,  # Final learning rate is max_lr / final_div_factor
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # OneCycleLR requires stepwise updates
                "frequency": 1,
            },
            "monitor": "val_loss",
        }


def train(
    checkpoint_path,
    checkpoint_save_dir,
    exp_name="clcl_pretrain",
    embed_dim=384,
    temperature=0.07,
    learning_rate=1e-4,
    weight_decay=1e-5,
    batch_size=32,
    num_workers=8,
    val_ratio=0.15,
    max_epochs=50,
    freeze_lidar_encoder=False,
    load_only_model=False,
    projection_type="linear",
):
    experiment_checkpoint_dir = os.path.join(checkpoint_save_dir, exp_name)
    os.makedirs(experiment_checkpoint_dir, exist_ok=True)

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

    train_dataset = A2D2Dataset(
        root_path="/homes/math/golombiewski/workspace/data/A2D2",
        val_ratio=val_ratio,
        split="train",
    )
    val_dataset = A2D2Dataset(
        root_path="/homes/math/golombiewski/workspace/data/A2D2",
        val_ratio=val_ratio,
        split="val",
    )

    train_loader = build_loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    model = CameraLidarPretrain(
        embed_dim=embed_dim,
        temperature=temperature,
        learning_rate=learning_rate,
        batch_size=batch_size,
        freeze_lidar_encoder=freeze_lidar_encoder,
        epoch_size=len(train_loader) / devices,
        projection_type=projection_type,
        weight_decay=weight_decay,
    )

    if checkpoint_path and load_only_model:
        model = CameraLidarPretrain.load_from_checkpoint(
            checkpoint_path,
            embed_dim=embed_dim,
            temperature=temperature,
            batch_size=batch_size,
            epoch_size=len(train_loader) / devices,
        )
        checkpoint_path = None
    elif not checkpoint_path:
        checkpoint_path = None

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
    early_stopping = EarlyStopping(monitor="val_loss", patience=20, mode="min", verbose=True)
    learningrate_callback = LearningRateMonitor(logging_interval="step")

    log_dir = "/homes/math/golombiewski/workspace/fast/clcl/logs"
    wandb_logger = WandbLogger(save_dir=log_dir, name=exp_name)
    csv_logger = CSVLogger(save_dir=log_dir, name=exp_name)

    trainer = pl.Trainer(
        # detect_anomaly=True,
        # fast_dev_run=10,
        logger=[wandb_logger, csv_logger],
        precision="32-true",
        accelerator=accelerator,
        devices=devices,
        limit_train_batches=None,
        max_epochs=max_epochs,
        strategy=strategy,
        callbacks=[checkpoint_callback, learningrate_callback, early_stopping],
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=checkpoint_path if not load_only_model else None,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Name of the experiment")
    parser.add_argument(
        "--checkpoint-save-dir",
        default="/homes/math/golombiewski/workspace/fast/clcl/checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--checkpoint", default="", help="Path to the checkpoint to load")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="Weight decay for training"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of workers for data loading"
    )
    parser.add_argument("--max-epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument(
        "--freeze-lidar-encoder", action="store_true", help="Freeze the lidar encoder"
    )
    parser.add_argument(
        "--load-only-model",
        action="store_true",
        help="Load only the model without training state",
    )
    parser.add_argument("--projection-type",
                        type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Type of projection head")
    args = parser.parse_args()
    assert args.name, "Empty name is not allowed"
    return args


if __name__ == "__main__":
    args = parse_args()
    train(
        checkpoint_path=args.checkpoint,
        exp_name=args.name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        max_epochs=args.max_epochs,
        freeze_lidar_encoder=args.freeze_lidar_encoder,
        load_only_model=args.load_only_model,
        checkpoint_save_dir=args.checkpoint_save_dir,
        projection_type=args.projection_type,
    )
