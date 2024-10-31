"""This module implements a MinkUNet point cloud encoder using MMDetection3D.
Adjust the config and checkpoint paths here, or provide them via a config file for flexibility.
"""
from typing import List
import torch
from torch import nn, Tensor
from mmdet3d.apis import init_model
from torch_scatter import scatter_mean
from model.point_cloud_encoder import PointCloudEncoder
from model.voxelize import voxelize

# Those usually don't change between once, so set them once here.
CONFIG_PATH = "/homes/math/golombiewski/workspace/clcl/src/model/minkunet34_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti.py"
CHECKPOINT_PATH = "/homes/math/golombiewski/workspace/clcl/src/model/minkunet34_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti_20230512_233511-bef6cad0.pth"

class MinkUNetEncoder(PointCloudEncoder):
    def __init__(self, embed_dim=384, freeze_encoder_weights=False, projection_type="linear"):
        super().__init__(embed_dim=embed_dim,
                         freeze_encoder_weights=freeze_encoder_weights,
                         projection_type=projection_type)
        self.config_path = CONFIG_PATH
        self.checkpoint_path = CHECKPOINT_PATH

        # Parameters from the MMDet3D MinkUNet model config
        self.voxel_params = {
            "voxel_type": "minkunet",
            "batch_first": False,
            "max_voxels": 80000,
            "voxel_layer": {
                "max_num_point_clouds": -1,
                "point_cloud_range": [-100, -100, -20, 100, 100, 20],
                "voxel_size": [0.05, 0.05, 0.05],
                "max_voxels": (-1, -1),
            },
        }

        self.model = init_model(self.config_path, self.checkpoint_path)
        self.maybe_freeze_encoder_weights()
        self.set_projection_type()

    def set_projection_type(self) -> None:
        """Set the projection head based on the projection type.
        Supported projection types are 'linear' and 'mlp'.
        """
        if self.projection_type not in ["linear", "mlp"]:
            raise ValueError("Invalid projection type. Use 'linear' or 'mlp'.")

        if self.projection_type == "linear":
            self.projection = nn.Linear(96, self.embed_dim)

        if self.projection_type == "mlp":
            self.projection = nn.Sequential(
                nn.Linear(96, 192),
                nn.GELU(),
                nn.Linear(192, self.embed_dim),
                nn.Dropout(0.1),
                nn.LayerNorm(self.embed_dim),
            )

    def forward(self, point_clouds: List[Tensor], batch_size: int = None) -> Tensor:
        """Generate embeddings from a list of point clouds by creating a voxel grid and feeding it
        into the MMDet3D MinkUNet encoder-decoder model. The model produces voxelwise features that
        are projected to the embedding space, then aggregated using a scattered mean operation.

        Args:
            point_clouds (List[torch.Tensor]): A list of torch tensors, each representing 
                a point cloud, where each tensor has shape (num_points, 4).
            batch_size (int, optional): The number of batches (i.e., point clouds) to process. 
                Defaults to len(point_clouds) if not specified.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, embed_dim) representing the embeddings
            for each point cloud.
        """
        # Compute batch size if not specified
        if batch_size is None:
            batch_size = len(point_clouds)

        voxel_dict = voxelize(point_clouds, self.voxel_params)
        input_dict = {"point_clouds": point_clouds, "voxels": voxel_dict}

        # Extract features from the MinkUNet encoder-decoder model that provides voxelwise features
        #  of dimension 96
        features = self.model.extract_feat(input_dict)  # (num_voxels, feature_dim == 96)

        # batch_indices tracks which voxel belongs to which poind cloud within the batch
        batch_indices = voxel_dict["coors"][:, -1].long()

        # The voxelwise features are first projected to the embedding space
        projected_features = self.projection(features)  # (num_voxels, embed_dim)

        # Then the scattered average pooling operation aggregates the voxelwise features into
        # global embeddings
        global_features = scatter_mean(
            projected_features, batch_indices, dim=0, dim_size=batch_size
        )  # (batch_size, embed_dim)

        return global_features


if __name__ == "__main__":
    # Sanity check
    lidar_encoder = MinkUNetEncoder().cuda()
    point_clouds = [torch.rand(1000, 4).cuda() for _ in range(12)]
    embeddings = lidar_encoder(point_clouds)

    # print("Output embeddings:", embeddings)
    # print("Output shape:", embeddings.shape)
