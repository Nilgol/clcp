import torch
from torch import nn
from mmdet3d.apis import init_model
from torch_scatter import scatter_mean
from voxelize import voxelize


class LidarEncoderMinkUNet(nn.Module):
    def __init__(self, embed_dim=384, freeze_encoder=False, projection_type="linear"):
        super().__init__()
        self.config_path = "/homes/math/golombiewski/workspace/mmdetection3d/configs/minkunet/minkunet34_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti.py"
        self.checkpoint_path = "/homes/math/golombiewski/workspace/clcl/model/minkunet34_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti_20230512_233511-bef6cad0.pth"
        self.voxel_params = {
            "voxel_type": "minkunet",
            "batch_first": False,
            "max_voxels": 80000,
            "voxel_layer": {
                "max_num_points": -1,
                "point_cloud_range": [-100, -100, -20, 100, 100, 20],
                "voxel_size": [0.05, 0.05, 0.05],
                "max_voxels": (-1, -1),
            },
        }
        self.model = init_model(self.config_path, self.checkpoint_path)
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
        if projection_type == "linear":
            self.projection = nn.Linear(96, embed_dim)
        elif projection_type == "mlp":
            self.projection = nn.Sequential(
                nn.Linear(96, 192),
                nn.GELU(),
                nn.Linear(192, embed_dim),
                nn.Dropout(0.1),
                nn.LayerNorm(embed_dim)
            )
        else:
            raise ValueError("Invalid projection type")

    def forward(self, points, batch_size=None):
        if batch_size is None:
            batch_size = len(points)

        voxel_dict = voxelize(points, self.voxel_params)
        input_dict = {"points": points, "voxels": voxel_dict}
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.model.extract_feat(input_dict)
        else:
            features = self.model.extract_feat(input_dict)  # (num_voxels, feature_dim)
        batch_indices = voxel_dict["coors"][:, -1].long()  # batch index for each voxel
        projected_features = self.projection(features)  # (num_voxels, embed_dim)
        print("projected feature shape:", projected_features.shape)
        global_features = scatter_mean(
            projected_features, batch_indices, dim=0, dim_size=batch_size
        )  # (batch_size, embed_dim)
        print("global feature shape:", global_features.shape)
        return global_features


if __name__ == "__main__":
    # Sanity check
    lidar_encoder = LidarEncoderMinkUNet().cuda()
    points = [torch.rand(1000, 4).cuda() for _ in range(1)]
    embeddings = lidar_encoder(points)

    print("Output shape:", embeddings.shape)