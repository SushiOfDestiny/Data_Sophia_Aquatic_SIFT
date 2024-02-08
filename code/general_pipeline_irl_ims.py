import subprocess

# List of scripts to run
scripts = ["general_pipeline_before_blender.py", "display_irl_matches.py"]

for script in scripts:
    subprocess.call(["python", script])

# alternaive solution: run following command in the bash terminal
# python imgs_preprocessing.py && python ...
