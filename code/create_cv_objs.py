import cv2 as cv
import numpy as np
import os
import sys

import descriptor as desc
import visu_hessian as vh
import visu_descriptor as visu_desc


matching_path = os.path.join(os.getcwd(), "../matching")
if os.path.exists(matching_path):
    sys.path.append(matching_path)
else:
    print(f"Directory {matching_path} does not exist")


from matching.saving import (
    save_keypoints,
    save_Dmatches,
    save_kp_pairs_to_arr,
)


from datetime import datetime

from numba import njit
import numba

# Goal is to convert into opencv objects such as keypoints and matches, the numpy arrays of distances and coordinates


def coords_to_kps(coords):
    """
    Convert a numpy array of coordinates to a list of OpenCV keypoints
    coords: np.array of shape (n, 2), of dtype int32
    """
    kp_list = []
    for id_coords in range(coords.shape[0]):
        x, y = coords[id_coords]
        # because opencv requires native python types, recast the coordinates to python types
        x, y = int(x), int(y)
        # print(f"types of coords: {type(x)}, {type(y)}")

        kp = cv.KeyPoint(x=x, y=y, size=1)
        kp_list.append(kp)
    return kp_list


def create_matches_list(distances_matches):
    """
    Create a list of OpenCV matches from a numpy array of distances
    distances_matches: np.array of shape (n, ), of dtype float32
    """
    matches_list = []
    for i in range(len(distances_matches)):
        match = cv.DMatch(_distance=distances_matches[i], _queryIdx=i, _trainIdx=i)
        matches_list.append(match)
    return matches_list


if __name__ == "__main__":
    photo_name = "rock_1"
    im_names = ["rock_1_left", "rock_1_right"]

    y_starts = [400, 400]
    y_lengths = [5, 5]
    x_starts = [800, 800]
    x_lengths = [5, 5]

    # load distances matches, and keypoints coordinates of matches
    storage_folder = "computed_distances"
    storage_filename_suffixes = ["dists", "coords_im1", "coords_im2"]
    filename_preffix = f"{photo_name}_y_{y_starts[0]}_{y_starts[1]}_{y_lengths[0]}_{y_lengths[1]}_{x_starts[0]}_{x_starts[1]}_{x_lengths[0]}_{x_lengths[1]}"
    loaded_objects = [None for i in range(3)]
    for id_obj in range(3):
        loaded_objects[id_obj] = np.load(
            f"{storage_folder}/{filename_preffix}_{storage_filename_suffixes[id_obj]}.npy"
        )

    # unpack the multiples lists
    distances_matches = loaded_objects[0]
    coords_ims_matches = loaded_objects[1:]  # shape = (2, n, 2)

    # create opencv lists of keypoints ordered like the distances
    kp_ims_matches = [
        coords_to_kps(coords_ims_matches[id_image]) for id_image in range(2)
    ]
    # create list of keypoints pairs, ordered like the distances
    kp_pairs = [
        (kp_ims_matches[0][i], kp_ims_matches[1][i])
        for i in range(len(kp_ims_matches[0]))
    ]

    # create opencv matches list
    matches_list = create_matches_list(distances_matches)

    print("finished creating opencv objects")

    print("beginning saving opencv objects to file")

    # save the matches and keypoints using function from matching/saving.py
    target_folder = "computed_matches"
    target_filename_preffix = f"{photo_name}_y_{y_starts[0]}_{y_starts[1]}_{y_lengths[0]}_{y_lengths[1]}_{x_starts[0]}_{x_starts[1]}_{x_lengths[0]}_{x_lengths[1]}"

    save_kp_pairs_to_arr(
        kp_pairs,
        f"{target_folder}/{target_filename_preffix}_kp_pairs",
    )
    print("finished saving kp_pairs")

    save_Dmatches(
        matches_list,
        f"{target_folder}/{target_filename_preffix}_matches.txt",
    )
    print("finished saving matches")

    for id_img in range(2):
        save_keypoints(
            kp_ims_matches[id_img],
            f"{target_folder}/{target_filename_preffix}_kp_{id_img}.txt",
        )
