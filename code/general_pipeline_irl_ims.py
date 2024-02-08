import subprocess
import logging

from computation_pipeline_hyper_params import *

from general_pipeline_before_blender import create_logger

# create the logger
logger = create_logger(path_to_log)

# List of scripts to run
scripts = ["general_pipeline_before_blender.py", "general_pipeline_after_blender.py"]

if __name__ == "__main__":
    result = subprocess.run(["python", scripts[0]], capture_output=True, text=True)
    logger.info(result.stdout)

    logger.info("Skip the Blender part")

    result = subprocess.run(["python", scripts[1]], capture_output=True, text=True)
    logger.info(result.stdout)


# alternaive solution: run following command in the bash terminal
# python imgs_preprocessing.py && python ...
