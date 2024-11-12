"""A script to find empty point clouds in the A2D2 dataset."""
import os
import glob
import pickle
import numpy as np
from tqdm import tqdm

# Set path to your A2D2 dataset directory
A2D2_ROOT_PATH = "/homes/math/golombiewski/workspace/data/A2D2"

def find_empty_point_clouds(data_dir, output_file):
    """Find empty point clouds in the A2D2 dataset."""
    empty_files = []

    lidar_paths = sorted(glob.glob(os.path.join(data_dir, "*/lidar/cam_front_center/*.npz")))
    for lidar_path in tqdm(lidar_paths, desc="Checking point clouds"):
        try:
            lidar_data = np.load(lidar_path)
            points = lidar_data["points"]
            if points.size == 0:
                empty_files.append(lidar_path)
        except Exception as e:
            print(f"Error loading {lidar_path}: {e}")

    with open(output_file, "wb") as f:
        pickle.dump(empty_files, f)

    print(f"Found {len(empty_files)} empty point cloud files.")


if __name__ == "__main__":
    root_path = (A2D2_ROOT_PATH)
    find_empty_point_clouds(root_path, "empty_point_clouds.pkl")
