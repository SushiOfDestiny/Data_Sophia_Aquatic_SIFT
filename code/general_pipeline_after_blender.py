import sys
import subprocess
from computation_pipeline_hyper_params import *
from filenames_creation import *
import argparse

from general_pipeline_before_blender import create_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sift", help="use SIFT instead of homemade descriptor", action="store_true")
    args = parser.parse_args()
    if args.sift:
        sift_radical = "_sift"
        log_filename = f"{dist_filename_prefix}_log{sift_radical}"
    else:
        sift_radical = ""
        log_filename = f"{dist_filename_prefix}_log"

    # create the logger
    logger = create_logger(f"{log_path}/{log_filename}.txt")


    scripts = [
        f"display{irl_radical}{sift_radical}_matches.py",
    ]   

    logger.info("Start running the post blender pipeline")

    for i in range(len(scripts)):
        logger.info(f"Run the script {scripts[i]}")

        result = subprocess.run(["python", scripts[i]], capture_output=True, text=True)
        logger.info(result.stdout)

        logger.info(f"Script {scripts[i]} is done.\n")

    logger.info("All scripts are done. The pipeline is finished.")
