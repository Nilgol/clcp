import os

# Define the source directory
root_dir = '/homes/math/golombiewski/workspace/test_stats'

# Loop over all files in the source directory
for file in os.listdir(root_dir):
    if file.endswith(".json"):
        # Construct the full path to the file
        old_file_path = os.path.join(root_dir, file)
        
        # Prepare the new filename
        # Split the filename to remove the timestamp (assuming it is the last segment before .json)
        parts = file.split('_')
        # Join parts excluding the timestamp part, which is second last before '.json'
        new_filename = '_'.join(parts[:-2]) + '.json'
        
        # Replace '000' with 'k' for better readability
        new_filename = new_filename.replace('000.', 'k.')
        
        # Construct the new file path
        new_file_path = os.path.join(root_dir, new_filename)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {old_file_path} to {new_file_path}")
