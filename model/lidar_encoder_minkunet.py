# lidar_encoder_minkunet.py
import torch
from torch import nn
from mmdet3d.apis import init_model
from voxelize import voxelize

class LidarEncoderMinkUNet(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        self.config_path = '/homes/math/golombiewski/workspace/mmdetection3d/configs/minkunet/minkunet34_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti.py'
        self.checkpoint_path = 'minkunet34_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti_20230512_233511-bef6cad0.pth'
        self.voxel_params = {
            'voxel_type': 'minkunet',
            'batch_first': False,
            'max_voxels': 80000,
            'voxel_layer': {
                'max_num_points': -1,
                'point_cloud_range': [-100, -100, -20, 100, 100, 20],
                'voxel_size': [0.05, 0.05, 0.05],
                'max_voxels': (-1, -1),
            }
        }
        self.model = init_model(self.config_path, self.checkpoint_path)
        self.projection = nn.Linear(96, embed_dim)  # Assuming output features have 96 dimensions

    def forward(self, points):
        voxel_dict = voxelize(points, self.voxel_params)
        input_dict = {
            'points': points,
            'voxels': voxel_dict
        }
        features = self.model.extract_feat(input_dict) # (num_voxels, feature_dim)
        projected_features = self.projection(features) # (num_voxels, embed_dim)
        embeddings = projected_features.mean(dim=0) # (embed_dim,)
        return embeddings

if __name__ == "__main__":
    # Sanity check
    lidar_encoder = LidarEncoderMinkUNet().cuda()
    points = [torch.rand(100, 4).cuda() for _ in range(4)]  # Batch size of 4, each with 100 points
    embeddings = lidar_encoder(points)

    print("Output embeddings:", embeddings)
    print("Output shape:", embeddings.shape)


