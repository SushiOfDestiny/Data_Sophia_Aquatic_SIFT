import subprocess
from computation_pipeline_hyper_params import blender_filename
from filenames_creation import original_imgs_path_prefix, sift_radical

if __name__ == '__main__':
    subprocess.call(["python", "general_pipeline_before_blender.py"])
    print(subprocess.check_output(f'blender {original_imgs_path_prefix}/{blender_filename}.blend -b --python-expr filter{sift_radical}_matches.py'))
    subprocess.call(["python", "general_pipeline_after_blender.py"])
