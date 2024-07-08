import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy

from data.a2d2_loader import build_loader
from model.lidar_encoder_minkunet import LidarEncoderMinkUNet
from model.image_encoder import ImageEncoder


class CameraLidarPretrain(pl.LightningModule):
    def __init__(self, embed_dim, temperature, batch_size, epoch_size):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.lidar_encoder = LidarEncoderMinkUNet(embed_dim=embed_dim)
        self.embed_dim = embed_dim
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.batch_size = batch_size
        self.epoch_size = epoch_size


    def contrastive_loss(self, image_embeddings, lidar_embeddings):
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        lidar_embeddings = F.normalize(lidar_embeddings, p=2, dim=-1)
        logits = (image_embeddings @ lidar_embeddings.T) / torch.exp(self.temperature)
        labels = torch.arange(image_embeddings.size(0), device=self.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_l = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_l) / 2

    def training_step(self, batch):
        images, point_clouds = batch
        image_embeddings = self.image_encoder(images)
        lidar_embeddings = self.lidar_encoder(point_clouds, self.batch_size)
        loss = self.contrastive_loss(image_embeddings, lidar_embeddings)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        self.log("temperature", self.temperature, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # Number of iterations for the first restart
            T_mult=1,  # A factor increases T_i after a restart
            eta_min=1e-6  # Minimum learning rate
        )
        return {"optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "train_loss"}


def train(
    checkpoint_path,
    checkpoint_save_dir,
    exp_name="clcl_pretrain",
    embed_dim=384,
    temperature=0.07,
    batch_size=32,
    num_workers=8,
    load_only_model=False,
):
    available_gpus = torch.cuda.device_count() or None
    accelerator = "gpu" if available_gpus else "cpu"
    devices = available_gpus if available_gpus else 1
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    train_loader = build_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    model = CameraLidarPretrain(
        embed_dim=embed_dim,
        temperature=temperature,
        batch_size=batch_size,
        epoch_size = len(train_loader) / devices
    )

    if checkpoint_path and load_only_model:
        model = CameraLidarPretrain.load_from_checkpoint(
            checkpoint_path,
            embed_dim=embed_dim,
            temperature=temperature,
            batch_size=batch_size,
            epoch_size=len(train_loader) / devices
        )
        checkpoint_path = None

    elif len(checkpoint_path) == 0:
        checkpoint_path = None

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_save_dir,
        save_top_k=-1,
        monitor="train_loss",
        save_last=True,
        # every_n_train_steps=250,
        save_on_train_epoch_end=True,
    )

    learningrate_callback = LearningRateMonitor(logging_interval="step")

    log_dir = '/homes/math/golombiewski/workspace/fast/clcl/logs'
    wandb_logger = WandbLogger(save_dir=log_dir, name=exp_name)
    csv_logger = CSVLogger(save_dir=log_dir, name=exp_name)

    trainer = pl.Trainer(
        # detect_anomaly=True,
        # fast_dev_run=20,
        logger=[wandb_logger, csv_logger],
        precision='32-true',
        accelerator=accelerator,
        devices=devices,
        limit_train_batches=None,
        max_epochs=50,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[checkpoint_callback, learningrate_callback],
    )

    trainer.fit(model=model, train_dataloaders=train_loader)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Name of the experiment")
    parser.add_argument("--checkpoint-save-dir", default='/homes/math/golombiewski/workspace/fast/clcl/checkpoints', help="Directory to save checkpoints")
    parser.add_argument("--checkpoint", required=False, default="", help="Path to the checkpoint to load")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--load-only-model", action="store_true", help="Load only the model without training state")
    args = parser.parse_args()
    assert args.name, "Empty name is not allowed"
    return args


if __name__ == "__main__":
    args = parse_args()
    train(
        checkpoint_path=args.checkpoint,
        exp_name=args.name,
        batch_size=args.batch_size,
        num_workers=args.workers,
        load_only_model=args.load_only_model,
        checkpoint_save_dir=args.checkpoint_save_dir,
    )