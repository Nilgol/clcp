import torch
import random

# Function to generate random point clouds with varying number of points
def generate_point_clouds(batch_size, min_points, max_points, feature_dim=96):
    point_clouds = []
    for _ in range(batch_size):
        num_points = random.randint(min_points, max_points)
        point_cloud = torch.randn(num_points, feature_dim)
        point_clouds.append(point_cloud)
    return point_clouds

def point_clouds_to_tensor(point_clouds):
    all_points = []
    point_cloud_indices = []
    
    for batch_idx, pc in enumerate(point_clouds):
        all_points.append(pc)
        point_cloud_indices.append(torch.full((pc.size(0), 1), batch_idx, dtype=torch.int64))
    
    # Concatenate all points and batch indices
    all_points = torch.cat(all_points, dim=0)  # Shape: (total_num_points, point_dim)
    point_cloud_indices = torch.cat(point_cloud_indices, dim=0)  # Shape: (total_num_points, 1)
    
    # Concatenate batch indices to the point features
    all_points_with_batch = torch.cat((point_cloud_indices, all_points), dim=1)  # Shape: (total_num_points, 1 + point_dim)
    
    return all_points_with_batch

def reshape_point_clouds(concatenated_tensor_with_index, point_cloud_indices, batch_size):
    # Remove the first column (index column)
    concatenated_tensor = concatenated_tensor_with_index[:, 1:]

    feature_dim = concatenated_tensor.size(1)
    max_points = max((point_cloud_indices == i).sum().item() for i in range(batch_size))
    
    reshaped_tensor = torch.zeros((batch_size, max_points, feature_dim), dtype=concatenated_tensor.dtype, device=concatenated_tensor.device)
    
    for i in range(batch_size):
        mask = (point_cloud_indices == i)
        num_points = mask.sum().item()
        reshaped_tensor[i, :num_points, :] = concatenated_tensor[mask]
    
    return reshaped_tensor

# Generate a batch of point clouds with 10 point clouds, each having between 100 and 150 points
batch_size = 4
min_points = 80
max_points = 100
point_clouds = generate_point_clouds(batch_size, min_points, max_points)

# Print the shapes of the generated point clouds
for i, pc in enumerate(point_clouds):
    print(f"Point cloud {i} shape: {pc.shape}")

concatenated_tensor = point_clouds_to_tensor(point_clouds)

print(f"Concatenated tensor shape: {concatenated_tensor.shape}")

reshaped_tensor = reshape_point_clouds(concatenated_tensor, concatenated_tensor[:,0].long(), batch_size)

print(f"Reshaped tensor shape: {reshaped_tensor.shape}")