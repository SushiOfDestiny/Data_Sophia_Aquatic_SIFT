import sys
import subprocess
from computation_pipeline_hyper_params import *
from filenames_creation import *

from general_pipeline_before_blender import create_logger


# create the logger
logger = create_logger(f"{log_path}/{log_filename}.txt")

if img_folder[:3] == "irl":
    scripts = [
        "display_irl_matches.py",
    ]
else:
    scripts = [
        f"display{sift_radical}_matches.py",
    ]

if __name__ == "__main__":

    logger.info("Start running the post blender pipeline")

    for i in range(len(scripts)):
        logger.info(f"Run the script {scripts[i]}")

        result = subprocess.run(["python", scripts[i]], capture_output=True, text=True)
        logger.info(result.stdout)

        logger.info(f"Script {scripts[i]} is done.\n")

    logger.info("All scripts are done. The pipeline is finished.")
