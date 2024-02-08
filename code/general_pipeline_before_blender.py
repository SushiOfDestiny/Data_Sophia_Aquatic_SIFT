import sys
import logging
import subprocess
from computation_pipeline_hyper_params import *
from filenames_creation import *


def create_logger(logpath):
    """
    Create a logger to save all prints and errors in a log file, located at logpath.
    """
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a file handler
    handler = logging.FileHandler(path_to_log, mode="a")
    handler.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    return logger


# create the logger
logger = create_logger(path_to_log)


# List of scripts to run
scripts = [
    "create_all_folders.py",
    "print_pipe_hparams.py",
    "imgs_preprocessing.py",
    "compute_desc_img.py",
    "compute_distances.py",
    "create_cv_objs.py",
]


if __name__ == "__main__":

    # first create all the folders
    subprocess.call(["python", scripts[0]])

    logger.info("Start running the pipeline")
    logger.info(f"All infos are saved in {path_to_log}\n")

    # run all the rest of the scripts
    for i in range(1, len(scripts)):
        logger.info(f"Run the script {scripts[i]}")

        result = subprocess.run(["python", scripts[i]], capture_output=True, text=True)
        logger.info(result.stdout)
        logger.info(f"Script {scripts[i]} is done.\n")

    logger.info(f"All scripts are done. Launch Blender or a display script.\n\n")

    # alternaive solution: run following command in the bash terminal
    # python imgs_preprocessing.py && python ...
