import os
import glob
import pickle
import numpy as np
from tqdm import tqdm

def find_empty_point_clouds(data_dir, output_file):
    empty_files = []

    lidar_paths = sorted(glob.glob(os.path.join(data_dir, '*/lidar/cam_front_center/*.npz')))
    for lidar_path in tqdm(lidar_paths, desc="Checking point clouds"):
        try:
            lidar_data = np.load(lidar_path)
            points = lidar_data['points']
            if points.size == 0:
                empty_files.append(lidar_path)
        except Exception as e:
            print(f"Error loading {lidar_path}: {e}")

    with open(output_file, 'wb') as f:
        pickle.dump(empty_files, f)

    print(f"Found {len(empty_files)} empty point cloud files.")

if __name__ == "__main__":
    root_path = '/homes/math/golombiewski/workspace/data/A2D2'  # Update this to your dataset path
    find_empty_point_clouds(root_path, 'empty_point_clouds.pkl')
