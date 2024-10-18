"""A PyTorch Lightning module for contrastive pretraining of pairs of camera images and lidar point clouds.
"""
import os
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from model.image_encoder import ImageEncoder
from model.point_cloud_encoder import PointCloudEncoder

class ImagePointCloudPretrain(LightningModule):
    """A module for multimodal self-supervised pretraining.
    The module takes embeddings from an image and point cloud encoder, respectively,
    and computes as contrastive loss between them.
    The module might be extended easily for other modalities or non-contrastive loss functions.

    Args:
        embed_dim (int): Dimension of the image and point cloud embeddings.
        temperature (float): Temperature parameter for the contrastive loss.
        batch_size (int): Batch size for training.
        optimizer_params (dict): Dictionary containing AdamW optimizer parameters.
        scheduler_params (dict): Dictionary containing learning rate scheduler parameters.
        freeze_point_cloud_encoder (bool): If True, freezes the point cloud encoder during training.
        projection_type (str): Type of projection head to use, either "linear" or "mlp".
    
    Attributes:
        image_encoder (ImageEncoder): Image encoder module.
        point_cloud_encoder (PointCloudEncoder): Point cloud encoder module.
        temperature (nn.Parameter): Temperature parameter for the contrastive loss.
        batch_size (int): Batch size for training.
        optimizer_params (dict): Dictionary containing AdamW optimizer parameters.
        scheduler_params (dict): Dictionary containing learning rate scheduler
    """
    def __init__(
        self,
        embed_dim,
        temperature,
        batch_size,
        optimizer_params,  # dictionary
        scheduler_params,  # dictionary
        freeze_point_cloud_encoder=False,
        projection_type="linear",
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.point_cloud_encoder = PointCloudEncoder(
            embed_dim=embed_dim,
            freeze_weights=freeze_point_cloud_encoder,
            projection_type=projection_type,
        )
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.batch_size = batch_size
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params

    def configure_optimizers(self):
        """Configure the optimizers and learning rate schedulers based on the provided parameters."""
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_params['learning_rate'],
            weight_decay=self.optimizer_params['weight_decay'],
            betas=self.optimizer_params['betas'],
        )

        warmup_length = self.scheduler_params['warmup']['total_iters']

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.scheduler_params['warmup']['start_factor'],
            total_iters=warmup_length
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.scheduler_params['cosine']['T_0'],
            T_mult=self.scheduler_params['cosine']['T_mult'],
            eta_min=self.scheduler_params['cosine']['eta_min'],
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_length],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def contrastive_loss(self, image_embeddings, lidar_embeddings):
        """Compute the contrastive loss of normalized image and point cloud embeddings."""
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        lidar_embeddings = F.normalize(lidar_embeddings, p=2, dim=-1)
        logits = (image_embeddings @ lidar_embeddings.T) / torch.exp(self.temperature)
        labels = torch.arange(image_embeddings.size(0), device=self.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_l = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_l) / 2

    def _common_step(self, batch):
        """Common step for training and validation."""
        images, point_clouds = batch
        image_embeddings = self.image_encoder(images)
        lidar_embeddings = self.point_cloud_encoder(point_clouds, self.batch_size)
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
