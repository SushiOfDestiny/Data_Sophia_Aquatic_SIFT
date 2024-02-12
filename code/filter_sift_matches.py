import bpy
import sys
import os
import logging
import numpy as np

sys.path.append(".")
from computation_pipeline_hyper_params import *

from filenames_creation import *


# add script folder to path, path corresponding to location of the virtual scene blender file
sys.path.append("../blender")

import matching as matching


if __name__ == "__main__":

    # Goal is to load the saved opencv matches and to filter them with the blender script
    print("----------------------------")
    cam_1 = bpy.data.objects["Cam_1"]
    cam_2 = bpy.data.objects["Cam_2"]

    storage_folder = f"{matches_path}"
    sift_kp_pairs_file = f"{storage_folder}/{kp_pairs_filename}{sift_suffix}.npy"

    # filter matches
    sift_correct_matches, sift_correct_matches_idxs, sift_matched_3d_pts = (
        matching.check_correct_match(sift_kp_pairs_file, cam_1, cam_2, epsilon)
    )

    print(len(sift_correct_matches), "Sift correct matches found")

    # save correct matches idxs
    sift_correct_matches_idxs_arr = np.array(
        sift_correct_matches_idxs, dtype=np.int32
    )  # first convert to numpy array

    matching.save_correct_matches(
        sift_correct_matches_idxs_arr,
        f"{storage_folder}/{correct_matches_idxs_filename}{sift_suffix}",
    )
