import os
import shutil

directories = ['SensorData', 'ExtractedData', 'PreprocessedData', 'Dataset', 'Tokens', 'Results']

for directory in directories:
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Removed existing directory: {directory}")
    
    os.makedirs(directory)
    print(f"Created directory: {directory}")
