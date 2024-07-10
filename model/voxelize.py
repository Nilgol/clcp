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
