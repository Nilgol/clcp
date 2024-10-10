import numpy as np
import torch
import torch.nn.functional as F


def ravel_hash(x):
    """Get voxel coordinates hash for np.unique."""
    assert x.ndim == 2, x.shape
    x = x - np.min(x, axis=0)
    x = x.astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0).astype(np.uint64) + 1

    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h


def sparse_quantize(coords, return_index=False, return_inverse=False):
    """Sparse Quantization for voxel coordinates used in Minkunet."""
    _, indices, inverse_indices = np.unique(
        ravel_hash(coords), return_index=True, return_inverse=True
    )
    coords = coords[indices]

    outputs = []
    if return_index:
        outputs += [indices]
    if return_inverse:
        outputs += [inverse_indices]
    return outputs


@torch.no_grad()
def voxelize(points, voxel_params):
    device = points[0].device  # Assuming all points are on the same device
    voxel_size = torch.tensor(voxel_params["voxel_layer"]["voxel_size"], device=device)
    point_cloud_range = torch.tensor(
        voxel_params["voxel_layer"]["point_cloud_range"], device=device
    )

    voxels, coors = [], []
    for i, res in enumerate(points):
        res_coors = torch.round(res[:, :3] / voxel_size).int()
        res_coors -= res_coors.min(0)[0]

        res_coors_numpy = res_coors.cpu().numpy()
        inds, point2voxel_map = sparse_quantize(
            res_coors_numpy, return_index=True, return_inverse=True
        )
        point2voxel_map = torch.from_numpy(point2voxel_map).to(device)
        if voxel_params["batch_first"]:
            res_voxel_coors = F.pad(res_coors[inds], (1, 0), mode="constant", value=i)
        else:
            res_voxel_coors = F.pad(res_coors[inds], (0, 1), mode="constant", value=i)

        res_voxels = res[inds]
        voxels.append(res_voxels)
        coors.append(res_voxel_coors)

    voxels = torch.cat(voxels, dim=0)
    coors = torch.cat(coors, dim=0)
    voxel_dict = {
        "voxels": voxels,
        "coors": coors,
        "point2voxel_map": point2voxel_map.long(),
    }
    return voxel_dict


# Testing script adjustments

import torch
from mmdet3d.apis import init_model

config_path = "/homes/math/golombiewski/workspace/mmdetection3d/configs/minkunet/minkunet34_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti.py"
checkpoint_path = "minkunet34_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti_20230512_233511-bef6cad0.pth"

model = init_model(config_path, checkpoint_path)

points = [torch.rand(100, 4).cuda() for _ in range(16)]

voxel_params = {
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

voxel_dict = voxelize(points, voxel_params)

input = {"points": points, "voxels": voxel_dict}

print("Points:")
print(points[0])
print("Voxels:")
print(voxel_dict["voxels"][0])

output = model(input)
features = model.extract_feat(input)

print("model output:")
print(output)
print("model features:")
print(model.extract_feat(input))
print("Shapes:")
print("model output shape:")
print(output.shape)
print("features shape:")
print(features.shape)
