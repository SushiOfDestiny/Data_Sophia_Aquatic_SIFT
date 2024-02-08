import bpy
import sys
import os
import time
import numpy as np

sys.path.append("../../../code")
from computation_pipeline_hyper_params import *

from filenames_creation import *


# add script folder to path, path corresponding to location of the virtual scene blender file
sys.path.append("../../../blender")

import matching as matching
import differential as differential

if __name__ == "__main__":
    # Goal is to load the saved opencv matches and to filter them with the blender script
    cam_1 = bpy.data.objects["Cam_1"]
    cam_2 = bpy.data.objects["Cam_2"]

    storage_folder = f"../../../code/{matches_path}"
    kp_pairs_file = f"{storage_folder}/{kp_pairs_filename}.npy"

    # filter matches
    correct_matches, correct_matches_idxs, matched_3d_pts = (
        matching.check_correct_match(kp_pairs_file, cam_1, cam_2, epsilon)
    )

    print(len(correct_matches), "correct matches found")

    # save correct matches idxs
    correct_matches_idxs_arr = np.array(
        correct_matches_idxs, dtype=np.int32
    )  # first convert to numpy array

    matching.save_correct_matches(
        correct_matches_idxs_arr,
        f"{storage_folder}/{correct_matches_idxs_filename}",
    )

    print("depth map computing...")
    t = time.time()
    dmap = differential.compute_depth_map(cam_1, bpy.context.scene)
    print(f"depth map computed in {time.time() - t}")
    np.save(f"{storage_folder}/dmap.npy", dmap)

    print(f"depth map saved to {storage_folder}/dmap.npy")


