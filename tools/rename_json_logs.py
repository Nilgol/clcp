import os

# Specify the directory containing the experiment subdirectories
base_dir = '/homes/math/golombiewski/workspace/work_dirs/segmenter_mask_acdc'

# Iterate over all items in the base directory
for item in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, item)
    # Check if the item is a directory
    if os.path.isdir(dir_path):
        # List all JSON files in the directory
        json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
        # There should be exactly one JSON file per directory
        if len(json_files) == 1:
            old_json_path = os.path.join(dir_path, json_files[0])
            new_json_name = f'{item}.json'
            new_json_path = os.path.join(dir_path, new_json_name)
            # Rename the JSON file
            os.rename(old_json_path, new_json_path)
            print(f'Renamed {old_json_path} to {new_json_path}')
        else:
            print(f'Error: Found {len(json_files)} JSON files in {dir_path}, expected 1.')
