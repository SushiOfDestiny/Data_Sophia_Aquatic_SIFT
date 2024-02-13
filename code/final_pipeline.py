import subprocess
import os
from computation_pipeline_hyper_params import *
from filenames_creation import *

if __name__ == '__main__':
    subprocess.call(["python", "general_pipeline_before_blender.py"])
    out = subprocess.check_output(f'blender {original_imgs_path_prefix}/{blender_filename}.blend -b -P filter_matches.py', shell=True)
    print(out)
    subprocess.call(["python", "general_pipeline_after_blender.py"])