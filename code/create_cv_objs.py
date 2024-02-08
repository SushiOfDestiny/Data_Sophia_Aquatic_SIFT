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

from computation_pipeline_hyper_params import *

from filenames_creation import *

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


def sort_matches_by_distance(matches_list):
    """
    Sort a list of matches by distance
    matches_list: list of cv.DMatch
    """
    matches_list.sort(key=lambda x: x.distance)
    return matches_list


if __name__ == "__main__":

    # load keypoints coordinates
    kp_coords = [
        np.load(f"{descrip_path}/{kp_coords_filenames[id_image]}.npy")
        for id_image in range(2)
    ]

    # load distances matches, and keypoints coordinates of matches
    distances_matches = np.load(f"{dist_path}/{dist_filename}.npy")
    matched_idx_ims = [
        np.load(f"{dist_path}/{matched_idx_filenames[id_image]}.npy")
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
    save_kp_pairs_to_arr(
        kp_pairs,
        f"{matches_path}/{kp_pairs_filename}",
    )
    print("finished saving kp_pairs")

    save_Dmatches(
        matches_list,
        f"{matches_path}/{matches_filename}.txt",
    )
    print("finished saving matches")

    for id_image in range(2):
        save_keypoints(
            kps_ims_objs[id_image],
            f"{matches_path}/{kp_filenames[id_image]}.txt",
        )
    print("finished saving keypoints")
