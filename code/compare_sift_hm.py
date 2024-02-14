# Testing pipeline after Blender script execution
import os
import sys

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import descriptor as desc
import visu_hessian as vh
import visu_descriptor as visu_desc

sys.path.append("../matching")
from saving import load_matches, load_keypoints

from computation_pipeline_hyper_params import *

from filenames_creation import *

import compute_desc_img as cp_desc

import display_matches as dm


def add_sift_radical(start, string):
    """
    Add the sift radical to a string after the start word
    start: str, the word after which to add the sift radical
    """
    # look for start in the string
    start_idx = string.find(start)
    # insert "_sift" after the start word
    return string[: start_idx + len(start)] + "_sift" + string[start_idx + len(start) :]


if __name__ == "__main__":
    # to launch without sift option activated, but sift equivalent computed

    # load images
    ims = [
        cv.imread(
            f"{relative_path}/{img_folder}/{im_names[id_image]}.{im_ext}",
            cv.IMREAD_GRAYSCALE,
        )
        for id_image in range(2)
    ]
    float_ims = np.load(f"{blurred_imgs_filename}.npy")

    # add sift to the raw_descrip_suffix
    sift_descrip_suffix = f"{raw_descrip_suffix}_sift{filt_radical}"

    # load unfiltered keypoints coordinates
    kps_coords = [
        np.load(f"{descrip_path}/{kp_coords_filenames}.npy") for id_image in range(2)
    ]

    # load unfiltered keypoints, matches and index of good matches
    kps = [
        load_keypoints(f"{matches_path}/{kp_filenames[id_image]}.txt")
        for id_image in range(2)
    ]
    matches = load_matches(f"{matches_path}/{matches_filename}.txt")

    # Load matches filtered by blender
    correct_matches_idxs = np.load(
        f"{matches_path}/{correct_matches_idxs_filename}.npy"
    )

    # load same objects but for sift
    # create filenames with sift radical in them
    start = raw_descrip_suffix[-5:-1]

    sift_kps_coords_filenames = [
        add_sift_radical(start, kp_coords_filenames[id_image]) for id_image in range(2)
    ]
    sift_kp_filenames = [
        add_sift_radical(start, kp_filenames[id_image]) for id_image in range(2)
    ]
    sift_matches_filename = add_sift_radical(start, matches_filename)

    sift_kps_coords = [
        np.load(f"{descrip_path}/{sift_kps_coords_filenames[id_image]}.npy")
        for id_image in range(2)
    ]
    sift_kps = [
        load_keypoints(f"{matches_path}/{sift_kp_filenames[id_image]}.txt")
        for id_image in range(2)
    ]
    sift_matches = load_matches(f"{matches_path}/{sift_matches_filename}.txt")
