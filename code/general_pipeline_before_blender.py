import subprocess

# List of scripts to run
scripts = [
    "imgs_preprocessing.py",
    "compute_desc_img.py",
    "compute_distances.py",
    "create_cv_objs.py",
]

for script in scripts:
    subprocess.call(["python", script])

# alternaive solution: run following command in the bash terminal
# python imgs_preprocessing.py && python ...
