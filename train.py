import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.a2d2_loader import build_loader
from model.lidar_encoder_minkunet import LidarEncoderMinkUNet
from model.image_encoder import ImageEncoder


class CameraLidarPretrain(pl.LightningModule):
    def __init__(self, embed_dim, batch_size, temperature=0.7):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.lidar_encoder = LidarEncoderMinkUNet(embed_dim=embed_dim)
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.temperature = nn.Parameter(torch.tensor(temperature))

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
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        steps_per_epoch = len(train_loader) // self.trainer.accumulate_grad_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,
            steps_per_epoch=steps_per_epoch,
            epochs=self.trainer.max_epochs,
            pct_start=0.1
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}


def train(
    checkpoint_path,
    checkpoint_save_dir,
    exp_name="clcl_pretrain",
    embed_dim=384,
    temperature=0.7,
    batch_size=32,
    num_workers=16,
    load_only_model=False,
):
    available_gpus = torch.cuda.device_count() or None
    accelerator = "gpu" if available_gpus else "cpu"
    devices = available_gpus if available_gpus else 1

    train_loader = build_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    model = CameraLidarPretrain(
        embed_dim=embed_dim,
        batch_size=batch_size,
        temperature=temperature,
    )

    if checkpoint_path and load_only_model:
        model = CameraLidarPretrain.load_from_checkpoint(
            checkpoint_path,
            embed_dim=embed_dim,
            batch_size=batch_size,
            temperature=temperature,
        )
        checkpoint_path = None

    elif len(checkpoint_path) == 0:
        checkpoint_path = None

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_save_dir,
        save_top_k=3,
        monitor="train_loss",
        save_last=True,
        every_n_train_steps=250,
        save_on_train_epoch_end=True,
    )

    learningrate_callback = LearningRateMonitor(logging_interval="step")

    logger = TensorBoardLogger("logs", name=exp_name)

    trainer = pl.Trainer(
        logger=logger,
        precision=16,
        accelerator=accelerator,
        devices=devices,
        limit_train_batches=None,
        max_epochs=1,
        strategy="ddp",
        callbacks=[checkpoint_callback, learningrate_callback],
        resume_from_checkpoint=checkpoint_path,
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