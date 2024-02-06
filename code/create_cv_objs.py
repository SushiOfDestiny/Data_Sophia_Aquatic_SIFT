import cv2 as cv
import numpy as np
import os
import sys

import descriptor as desc
import visu_hessian as vh
import visu_descriptor as visu_desc


sys.path.append("../matching")
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


def create_matches_list(distances_matches, matched_idx_ims):
    """
    Create a list of OpenCV matches from a numpy array of distances
    distances_matches: numpy array of shape (n, ), of dtype float32
    matched_idx_ims: list of 2 np.array of shape (n, )
    """
    matches_list = []
    for i in range(len(distances_matches)):
        match = cv.DMatch(
            _distance=distances_matches[i],
            _queryIdx=matched_idx_ims[0][i],
            _trainIdx=matched_idx_ims[1][i],
        )
        matches_list.append(match)
    return matches_list


if __name__ == "__main__":
    photo_name = "rock_1"
    im_names = ["rock_1_left", "rock_1_right"]

    # set the coordinates of the subimages
    y_starts = [386, 459]
    y_lengths = [10, 10]
    x_starts = [803, 806]
    x_lengths = [20, 20]

    # load keypoints coordinates
    kp_coords_filename_prefixes = [
        f"{im_names[id_image]}_y_{y_starts[id_image]}_{y_lengths[id_image]}_x_{x_starts[id_image]}_{x_lengths[id_image]}"
        for id_image in range(2)
    ]
    kp_coords = [
        np.load(
            f"computed_descriptors/{kp_coords_filename_prefixes[id_image]}_coords.npy"
        )
        for id_image in range(2)
    ]

    # load distances matches, and keypoints coordinates of matches
    matched_filename_prefix = f"{photo_name}_y_{y_starts[0]}_{y_starts[1]}_{y_lengths[0]}_{y_lengths[1]}_x_{x_starts[0]}_{x_starts[1]}_{x_lengths[0]}_{x_lengths[1]}"

    distances_matches = np.load(
        f"computed_distances/{matched_filename_prefix}_dists.npy"
    )
    matched_idx_ims = [
        np.load(
            f"computed_distances/{matched_filename_prefix}_matched_idx_im{id_image+1}.npy"
        )
        for id_image in range(2)
    ]

    # create opencv lists of original keypoints
    # kps_ims_objs[id_image][k, l] is the l-th coordinate of the k-th keypoint of image id_image
    kps_ims_objs = [coords_to_kps(kp_coords[id_image]) for id_image in range(2)]

    # create list of keypoints pairs, ordered like the distances
    kp_pairs = [
        (
            kps_ims_objs[0][matched_idx_ims[0][id_dist]],
            kps_ims_objs[1][matched_idx_ims[1][id_dist]],
        )
        for id_dist in range(len(distances_matches))
    ]

    # create opencv matches list
    matches_list = create_matches_list(distances_matches, matched_idx_ims)

    print("finished creating opencv objects")

    print("beginning saving opencv objects to file")

    # save the matches and keypoints using function from matching/saving.py
    target_matched_filename_prefix = f"{photo_name}_y_{y_starts[0]}_{y_starts[1]}_{y_lengths[0]}_{y_lengths[1]}_x_{x_starts[0]}_{x_starts[1]}_{x_lengths[0]}_{x_lengths[1]}"

    save_kp_pairs_to_arr(
        kp_pairs,
        f"computed_matches/{target_matched_filename_prefix}_kp_pairs_arr",
    )
    print("finished saving kp_pairs")

    save_Dmatches(
        matches_list,
        f"computed_matches/{target_matched_filename_prefix}_matches.txt",
    )
    print("finished saving matches")

    for id_img in range(2):
        save_keypoints(
            kps_ims_objs[id_img],
            f"computed_matches/{target_matched_filename_prefix}_kp_{id_img}.txt",
        )
