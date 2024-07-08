import os
import glob
import numpy as np
import pickle

def check_npz_files(root_path, keys=['points', 'reflectance', 'row', 'col']):
    npz_files = glob.glob(os.path.join(root_path, '*/lidar/cam_front_center/*.npz'))
    missing_keys_files = {}

    for npz_file in npz_files:
        data = np.load(npz_file)
        missing_keys = [key for key in keys if key not in data]
        
        if missing_keys:
            print(f"File {npz_file} is missing keys: {missing_keys}")
            missing_keys_files[npz_file] = missing_keys

    return missing_keys_files

def save_missing_keys_files(missing_keys_files, output_path='missing_keys_files.pkl'):
    with open(output_path, 'wb') as f:
        pickle.dump(missing_keys_files, f)

if __name__ == "__main__":
    root_path = '/homes/math/golombiewski/workspace/data/A2D2'
    missing_keys_files = check_npz_files(root_path)
    save_missing_keys_files(missing_keys_files)
    print(f"Saved missing keys file information to missing_keys_files.pkl")
