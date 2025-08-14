import os
import shutil

# Directories to reset
directories = ['SensorData', 'ExtractedData', 'PreprocessedData', 'Dataset', 'Tokens', 'Results']

for directory in directories:
    # Remove the directory if it exists
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Removed existing directory: {directory}")
    
    # Recreate the directory
    os.makedirs(directory)
    print(f"Created directory: {directory}")
