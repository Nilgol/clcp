import os

def rename_files(directory, old_substr, new_substr):
    # Loop through all the files in the directory
    for filename in os.listdir(directory):
        if old_substr in filename:
            # Create the new filename by replacing the old substring with the new one
            new_filename = filename.replace(old_substr, new_substr)
            # Construct the full file paths
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed '{old_file_path}' to '{new_file_path}'")

# Usage
directory = '/homes/math/golombiewski/workspace/test_stats'
old_substr = 'frozen.'
new_substr = 'frozenbb.'
rename_files(directory, old_substr, new_substr)
