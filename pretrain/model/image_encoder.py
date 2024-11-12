"""A module that contains the abstract base class for all image encoders."""

from torch import nn
from abc import ABC, abstractmethod


class ImageEncoder(ABC, nn.Module):
    """
    Abstract base class for all image encoders.

    Args:
        embed_dim (int): The dimension of the output embeddings.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(self, x):
        """Abstract method for generating embeddings from images.
        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The image embeddings.
        """
        pass
