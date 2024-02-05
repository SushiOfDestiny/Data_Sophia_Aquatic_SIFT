import bpy
import sys
import os
import numpy as np

print(os.getcwd())

# add script folder to path, path corresponding to location of the virtual scene blender file
sys.path.append("../../../blender")

import matching as matching

if __name__ == "__main__":
    # Goal is to load the saved opencv matches and to filter them with the blender script
    print("----------------------------")
    cam_1 = bpy.data.objects["Cam_1"]
    cam_2 = bpy.data.objects["Cam_2"]

    photo_name = "rock_1"
    im_names = ["rock_1_left", "rock_1_right"]

    y_starts = [400, 400]
    y_lengths = [5, 5]
    x_starts = [800, 800]
    x_lengths = [5, 5]

    storage_folder = "../../../code/computed_matches"
    filename_prefix = f"{photo_name}_y_{y_starts[0]}_{y_starts[1]}_{y_lengths[0]}_{y_lengths[1]}_{x_starts[0]}_{x_starts[1]}_{x_lengths[0]}_{x_lengths[1]}"

    kp_pairs_file = f"{storage_folder}/{filename_prefix}_kp_pairs.npy"

    # define filtering precision threshold
    epsilon = 1e-3

    # filter matches
    correct_matches, correct_matches_idxs, matched_3d_pts = (
        matching.check_correct_match(kp_pairs_file, cam_1, cam_2, epsilon)
    )

    # save correct matches idxs
    correct_matches_idxs_arr = np.array(
        correct_matches_idxs, dtype=np.int32
    )  # first convert to numpy array

    correct_matches_idxs_filename = f"{storage_folder}/{filename_prefix}_idxs"
    matching.save_correct_matches(
        correct_matches_idxs_arr, correct_matches_idxs_filename
    )
