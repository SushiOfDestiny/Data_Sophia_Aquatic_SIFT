import os
from computation_pipeline_hyper_params import *
from filenames_creation import *

# Goal: if the folders in which computed objects do not exist, create them

if __name__ == "__main__":
    for folder_name in required_folders_paths:
        try:
            os.makedirs(folder_name)
        except FileExistsError:
            print(f"Folder {folder_name} already exists")
            pass
        else:
            print(f"Folder {folder_name} created")
            pass
