import torch
from torch import nn
from mmdet3d.apis import init_model
import torchsparse
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
        self.embed_dim = embed_dim
        self.model = init_model(self.config_path, self.checkpoint_path)

        del self.model.backbone.decoder

        # Define a new forward method for the backbone to bypass accessing the deleted decoder
        def encoder_only_forward(
            self, voxel_features: torch.Tensor, coors: torch.Tensor
        ) -> torch.Tensor:
            print("Input voxel features shape:", voxel_features.shape)
            x = torchsparse.SparseTensor(voxel_features, coors)
            print("Initial sparse tensor shape:", x.F.shape)
            x = self.conv_input(x)
            print("After initial conv_input shape:", x.F.shape)
            for i, encoder_layer in enumerate(self.encoder):
                x = encoder_layer(x)
                print(f"After encoder layer {i} shape:", x.F.shape)
            return x.F  # extracts the dense feature tensor from the sparse tensor

        self.model.backbone.forward = encoder_only_forward.__get__(self.model.backbone)

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        if projection_type == "linear":
            self.projection = nn.Linear(256, self.embed_dim)
        elif projection_type == "mlp":
            self.projection = nn.Sequential(
                nn.Linear(256, 320),
                nn.GELU(),
                nn.Linear(320, self.embed_dim),
                nn.Dropout(0.2),
                nn.LayerNorm(self.embed_dim),
            )
        else:
            raise ValueError("Invalid projection type")

    def forward(self, points, batch_size=None):
        if batch_size is None:
            batch_size = len(points)
        # print("batch size:", batch_size)

        voxel_dict = voxelize(points, self.voxel_params)
        # print("voxels:", voxel_dict['voxels'].shape)
        input_dict = {"points": points, "voxels": voxel_dict}
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.model.extract_feat(input_dict)
        else:
            features = self.model.extract_feat(input_dict)  # (num_voxels, feature_dim)
        batch_indices = voxel_dict["coors"][:, -1].long()  # batch index for each voxel
        # print("features shape:", features.shape)
        projected_features = self.projection(features)  # (num_voxels, self.embed_dim)
        # print("projected feature shape:", projected_features.shape)
        global_features = scatter_mean(
            projected_features, batch_indices, dim=0, dim_size=batch_size
        )
        # print("global feature shape:", global_features.shape)
        return global_features


if __name__ == "__main__":
    # Sanity check
    lidar_encoder = LidarEncoderMinkUNet().cuda()
    points = [
        torch.rand(100, 4).cuda(),  # Small point cloud
        torch.rand(5000, 4).cuda(),  # Medium point cloud
        torch.rand(20000, 4).cuda(),  # Large point cloud
        torch.rand(500, 4).cuda(),  # Small point cloud
        torch.rand(12000, 4).cuda(),  # Large point cloud
        torch.rand(50000, 4).cuda(),
        torch.rand(10, 4).cuda(),
        torch.rand(2, 4).cuda(),
        torch.rand(120000, 4).cuda(),
    ]
    embeddings = lidar_encoder(points)
    # print("Output shape:", embeddings.shape)
