import subprocess
from computation_pipeline_hyper_params import *
from filenames_creation import original_imgs_path_prefix
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hmd", help="launch homemade descriptor pipeline", action="store_true")
    parser.add_argument("--sift", help="launch SIFT pipeline", action="store_true")
    args = parser.parse_args()

    if args.sift:
        subprocess.call(["python", "general_pipeline_before_blender.py", "--sift"])
        out = subprocess.check_output(f'blender {original_imgs_path_prefix}/{blender_filename}.blend -b -P filter_sift_matches.py', shell=True)
        print(out)
        subprocess.call(["python", "general_pipeline_after_blender.py", "--sift"])

    if args.hmd:
        subprocess.call(["python", "general_pipeline_before_blender.py"])
        out = subprocess.check_output(f'blender {original_imgs_path_prefix}/{blender_filename}.blend -b -P filter_matches.py', shell=True)
        print(out)
        subprocess.call(["python", "general_pipeline_after_blender.py"])
