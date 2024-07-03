import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from data.a2d2_loader import build_loader
from model.lidar_encoder_minkunet import LidarEncoderMinkUNet
from model.image_encoder import ImageEncoder


class CameraLidarPretrain(pl.LightningModule):
    def __init__(self, lidar_encoder, image_encoder, embedding_dimension, batch_size, temperature=0.07):
        super().__init__()
        self.lidar_encoder = lidar_encoder
        self.image_encoder = image_encoder
        self.embedding_dimension = embedding_dimension
        self.batch_size = batch_size
        self.temperature = temperature

    def contrastive_loss(self, image_embeddings, lidar_embeddings, temperature):
        # Normalize the embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        lidar_embeddings = F.normalize(lidar_embeddings, p=2, dim=-1)

        # Compute cosine similarity matrix
        logits = (image_embeddings @ lidar_embeddings.T) / temperature

        # Labels are the indices of the correct pairs
        labels = torch.arange(image_embeddings.size(0), device=self.device)

        # Cross entropy loss for both directions
        loss_i = F.cross_entropy(logits, labels)
        loss_l = F.cross_entropy(logits.T, labels)
        
        return (loss_i + loss_l) / 2

    def training_step(self, batch, batch_idx):
        images, point_clouds = batch
        image_embeddings = self.image_encoder(images)
        lidar_embeddings = self.lidar_encoder(point_clouds)
        loss = self.contrastive_loss(image_embeddings, lidar_embeddings, self.temperature)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader()) // self.trainer.accumulate_grad_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,
            steps_per_epoch=steps_per_epoch,
            epochs=self.trainer.max_epochs,
            pct_start=0.1
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}


def train(
    data_dir,
    name,
    checkpoint_save_dir,
    checkpoint_path,
    batch_size,
    num_workers,
    load_only_model=False,
    loss_function="mse",
    a2d2_root_path='/homes/math/golombiewski/workspace/data/A2D2',
    a2d2_config_path='/homes/math/golombiewski/workspace/data/A2D2_general/cams_lidars.json',
    dataset_name="a2d2",
    embedding_dimension=self.embedding_dimension
):
    """Train the model."""
    image_encoder = ImageEncoder(embed_dim=embedding_dimension)
    lidar_encoder = LidarEncoderMinkUNet(embed_dim=embedding_dimension)

    available_gpus = torch.cuda.device_count() or None
    accelerator = "gpu" if available_gpus else "cpu"
    devices = available_gpus if available_gpus else 1

    train_loader = build_loader(
        data_dir,
        # clip_preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        a2d2_root_path=a2d2_root_path,
        a2d2_config_path=a2d2_config_path,
        dataset_name=dataset_name,
    )

    model = CameraLidarPretrain(
        lidar_encoder,
        image_encoder,
        embedding_dimension,
        batch_size,
        len(train_loader) / devices,
        loss_function
    )

    if len(checkpoint_path) and load_only_model:
        model = CameraLidarPretrain.load_from_checkpoint(
            checkpoint_path,
            lidar_encoder=lidar_encoder,
            image_encoder=image_encoder,
            batch_size=batch_size,
            epoch_size=len(train_loader) / devices,
            loss=loss_function,
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

    trainer = pl.Trainer(
        precision=16,
        accelerator=accelerator,
        devices=devices,
        limit_train_batches=None,
        max_epochs=1,
        # logger=wandb_logger,
        strategy="ddp",
        callbacks=[checkpoint_callback, learningrate_callback],
        resume_from_checkpoint=checkpoint_path,
    )

    trainer.fit(model=model, train_dataloaders=train_loader)

def parse_args():
    pass ## Add my own arguments here
    return args


if __name__ == "__main__":
    args = parse_args()
    train(
        ## Add my own arguments here
    )
