import os

# Path to the folder
folder_path = './path/to/folder'

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file name ends with ' copy'
    if filename.endswith(' copy'):
        # Generate the new file name by removing ' copy'
        new_filename = filename.replace(' copy', '')
        # Get the full paths for the old and new file names
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)
        # Rename the file
        os.rename(old_filepath, new_filepath)
        print(f'Renamed: {filename} -> {new_filename}')

print('All files have been renamed.')
