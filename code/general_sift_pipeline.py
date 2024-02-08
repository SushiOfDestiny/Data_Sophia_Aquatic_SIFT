import subprocess

# computation will be ran on current pipeline image

# List of scripts to run
scripts = [
    "compute_sift_kps.py" "display_sift_matches.py",
]

for script in scripts:
    subprocess.call(["python", script])
