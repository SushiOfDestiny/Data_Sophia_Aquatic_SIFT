import sys
import subprocess
from computation_pipeline_hyper_params import *
from filenames_creation import *

from general_pipeline_before_blender import create_logger


if __name__ == "__main__":

    # create the logger
    logger = create_logger(f"{log_path}/{log_filename}.txt")

    # scripts = [
    #     f"display{irl_radical}{sift_radical}_matches.py",
    # ]

    scripts = [
        "print_pipe_hparams.py",
        f"display{irl_radical}_matches.py",
    ]

    logger.info("Start running the post blender pipeline")

    for i in range(len(scripts)):
        logger.info(f"Run the script {scripts[i]}")

        result = subprocess.run(["python", scripts[i]], capture_output=True, text=True)
        logger.info(result.stdout)

        logger.info(f"Script {scripts[i]} is done.\n")

    logger.info("All scripts are done. The pipeline is finished.")
