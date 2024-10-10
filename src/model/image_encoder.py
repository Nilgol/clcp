from torch import nn
import timm


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        self.model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0,
        )
        self.projection = nn.Linear(self.model.num_features, embed_dim)

    def forward(self, x):
        features = self.model(x)  # Extract features using the ViT model
        embeddings = self.projection(
            features
        )  # Project features to the desired embedding dimension
        return embeddings


if __name__ == "__main__":
    # Quick sanity check

    import torch

    image_encoder = ImageEncoder()
    dummy_batch = torch.randn(12, 3, 224, 224)
    output = image_encoder(dummy_batch)
    print("Output shape:", output.shape)
