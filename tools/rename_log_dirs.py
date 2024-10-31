import os

# Define the base directory containing the log subdirectories
base_dir = '/homes/math/golombiewski/workspace/work_dirs/segmenter_mask_acdc'

# Iterate over all items in the base directory
for item in os.listdir(base_dir):
    subdir = os.path.join(base_dir, item)
    if os.path.isdir(subdir):  # Check if the item is a directory
        config_path = os.path.join(subdir, 'vis_data', 'config.py')
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                for line in file:
                    if line.strip().startswith('load_from'):
                        # Find the line containing the path
                        path = line.split('=')[1].strip().strip("'")
                        # Split the path and find the experiment name
                        parts = path.split('/')
                        experiment_name_index = parts.index('work_dirs') + 1
                        experiment_name = parts[experiment_name_index]
                        iter_name = parts[-1].split('.')[0]  # Get the filename without extension
                        # Create new directory name by prepending the experiment name and iteration
                        new_dir_name = f"{experiment_name}_{iter_name}_{item}"
                        new_dir_path = os.path.join(base_dir, new_dir_name)
                        print(new_dir_path)
                        # Rename the directory
                        os.rename(subdir, new_dir_path)
                        print(f"Renamed '{subdir}' to '{new_dir_path}'")
                        break
