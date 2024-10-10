import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from model.image_encoder import ImageEncoder
from model.lidar_encoder_minkunet import LidarEncoderMinkUNet

class CameraLidarPretrain(pl.LightningModule):
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
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        self.log("temperature", self.temperature, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1, T_mult=2, eta_min=1e-5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
