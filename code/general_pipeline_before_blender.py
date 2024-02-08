import subprocess
import sys
from computation_pipeline_hyper_params import *
from filenames_creation import *

if __name__ == "__main__":
    # List of scripts to run
    scripts = [
        "create_all_folders.py",
        "imgs_preprocessing.py",
        "compute_desc_img.py",
        "compute_distances.py",
        "create_cv_objs.py",
    ]

    for script in scripts:
        subprocess.call(["python", script])

    # alternaive solution: run following command in the bash terminal
    # python imgs_preprocessing.py && python ...
