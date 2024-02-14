import subprocess
import logging

from computation_pipeline_hyper_params import *

from filenames_creation import *

from general_pipeline_before_blender import create_logger



# List of scripts to run
scripts = ["general_pipeline_before_blender.py", "display_irl_matches.py"]

if __name__ == "__main__":
    # create the logger
    logger = create_logger(f"{log_path}/{log_filename}.txt")


    result = subprocess.run(["python", scripts[0]], capture_output=True, text=True)
    logger.info(result.stdout)

    logger.info("Skip the Blender part")

    result = subprocess.run(["python", scripts[1]], capture_output=True, text=True)
    logger.info(result.stdout)


# alternaive solution: run following command in the bash terminal
# python imgs_preprocessing.py && python ...
