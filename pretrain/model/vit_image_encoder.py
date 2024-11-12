"""A module that contains a small ViT image encoder class."""
from torch import nn
import timm
from model.image_encoder import ImageEncoder


class VitImageEncoder(ImageEncoder):
    """Class for a small Vision Transformer (ViT) image encoder from PyTorch Image Models (timm).
    
    Args:
        embed_dim (int): The dimension of the output embeddings.
    
    Attributes:
        model (nn.Module): The Vision Transformer model.
        projection (nn.Linear): A linear projection to the embedding space.
    """
    def __init__(self, embed_dim=384):
        super().__init__(embed_dim=embed_dim)
        self.model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0,
        )
        self.projection = nn.Linear(self.model.num_features, embed_dim)

    def forward(self, x):
        """Generate embeddings from an image by passing it through the ViT model and projecting the
        features to the embedding space."""
        features = self.model(x)
        embeddings = self.projection(features)
        return embeddings


if __name__ == "__main__":
    # Quick sanity check
    import torch

    image_encoder = VitImageEncoder()
    dummy_batch = torch.randn(12, 3, 224, 224)
    output = image_encoder(dummy_batch)
    print("Output shape:", output.shape)
