"""This module supplies an abstract base class for point cloud encoders."""
from abc import ABC, abstractmethod
from typing import List
from torch import nn, Tensor


class PointCloudEncoder(ABC, nn.Module):
    """
    Abstract base class for all point cloud encoders.

    It is advised to use this class as base class for all point cloud encoders.
    All subclasses must implement the forward and set_projection_type methods.

    The forward method should take a list of point clouds in tensor format and return embeddings 
    as a tensor of shape (batch_size, embed_dim).

    The set_projection_type method should set the projection head based on the projection type.
    Possible types are, for example, "linear" or "mlp".

    Subclasses should ensure that the encoder weights (but not the projection weights) are frozen
    if the `freeze_encoder_weights` flag is set to True.
    For this, the base class supplies the maybe_freeze_weights method. This method should be called
    after self.freeze_weights is set and the encoder model is initialized, but before the
    projection head is set.
    
    Args:
        embed_dim (int): The dimension of the output embeddings.
        freeze_encoder_weights (bool): If True, freeze the encoder weights during training.
        projection_type (str): Type of projection to map encoder outputs to embeddings.

    Attributes:
        embed_dim (int): The dimension of the output embeddings.
        freeze_encoder_weights (bool): If True, freeze the encoder weights during training.
        projection_type (str): Type of projection to map encoder outputs to embeddings.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        freeze_encoder_weights: bool = False,
        projection_type: str = "linear",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.freeze_encoder_weights = freeze_encoder_weights
        self.projection_type = projection_type

    @abstractmethod
    def forward(self, point_clouds: List[Tensor], batch_size: int = None) -> Tensor:
        """
        Abstract method for generating embeddings from point clouds.

        Each subclass must implement this. The method should take a list of point clouds and return
        embeddings as a tensor of shape (batch_size, embed_dim).

        Args:
            point_clouds (List[torch.Tensor]): A list of torch.Tensors, each representing a point cloud.
            batch_size (int, optional): The batch size, typically len(point_clouds).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, embed_dim) representing the embeddings.
        """

    @abstractmethod
    def set_projection_type(self) -> None:
        """
        Abstract method to set the projection type for mapping encoder outputs to embeddings.

        Subclasses must implement this method to set the projection head based on projection type.
        It is advised to raise an error if the supplied projection type is not supported.

        Raises:
            ValueError: If an unsupported projection type is provided.
        """

    def maybe_freeze_encoder_weights(self):
        """Method to freeze encoder weights if the `freeze_encoder_weights` flag is set to True.

        This method should be called after initializing the encoder model, but before setting the
        projection.

        Raises:
            RuntimeError: If the encoder is not initialized or if the projection layer is already set.
        """
        if not hasattr(self, "model"):
            raise RuntimeError(
                "Encoder model not initialized. Call maybe_freeze_encoder_weights() after loading the\
                    model."
            )

        if any("projection" in name for name, _ in self.named_parameters()):
            raise RuntimeError(
                "Projection layer is already defined. Call maybe_freeze_encoder_weights before setting\
                    the projection layer."
            )

        if self.freeze_encoder_weights:
            for param in self.model.parameters():
                param.requires_grad = False
