import cv2 as cv
import numpy as np
import os
import sys

import descriptor as desc
import visu_hessian as vh
import visu_descriptor as visu_desc

from datetime import datetime

from numba import njit
import numba

from tqdm import tqdm

from computation_pipeline_hyper_params import *

from filenames_creation import *


# Goal: load descriptors arrays, each for 1 image, flatten them and compute the distances between them
# first use bruteforce matching


@njit
def compute_distances_matches_pairs(subimage_descriptors, y_lengths, x_lengths):
    """
    Compute distances_matches between all pairs of descriptors from 2 images
    subimage_descriptors: list of 2 arrays of descriptors, each for 1 image, each containing as much element as pixels, each element
    of shape (3 * nb_bins * nb_bins * nb_angular_bins, )
    return:
    - 1D numpy array of distances_matches, with distances_matches[id_pix1 * nb_pix2 + id_pix2] = distance between pixel
    of coords subimage_coords[0][id_pix1] of image 1 and pixel subimage_coords[2][id_pix2] of image 2
    - 1D numpy array of indices in subimage_coords[0] of the pixel in image 1 that appears in the match in distances_matches, at the same position
    - 1D numpy array of indices in subimage_coords[1] of the pixel in image 2 that appears in the match in distances_matches, at the same position
    """

    # initialize null array of distances_matches
    nb_matches = y_lengths[0] * x_lengths[0] * y_lengths[1] * x_lengths[1]
    distances_matches = np.zeros((nb_matches,), dtype=np.float32)
    idx1_matches = np.zeros((nb_matches,), dtype=np.int32)
    idx2_matches = np.zeros((nb_matches,), dtype=np.int32)

    for idx_pixel_im1 in tqdm(range(len(subimage_descriptors[0]))):

        descrip_pixel_im1 = subimage_descriptors[0][idx_pixel_im1]

        for idx_pixel_im2 in range(len(subimage_descriptors[1])):

            descrip_pixel_im2 = subimage_descriptors[1][idx_pixel_im2]

            # compute index of distance in the distances_matches array
            dist_idx = idx_pixel_im1 * len(subimage_descriptors[1]) + idx_pixel_im2
            distances_matches[dist_idx] = desc.compute_descriptor_distance(
                descrip_pixel_im1, descrip_pixel_im2
            )

            # store coordinates
            idx1_matches[dist_idx] = idx_pixel_im1
            idx2_matches[dist_idx] = idx_pixel_im2

    return distances_matches, idx1_matches, idx2_matches


@njit(parallel=True)
def compute_minimal_distances_matches_pairs(subimage_descriptors, y_lengths, x_lengths):
    """
    Compute matches of minimal distances to pixels of image 1
    subimage_descriptors: list of 2 arrays of descriptors, each for 1 image, each containing as much element as pixels, each element
    of shape (3 * nb_bins * nb_bins * nb_angular_bins, )
    return:
    - 1D numpy array of distances_matches, with distances_matches[id_pix1] = minimal distance between pixel of image1 at index id_pix1 and any pixel of image 2
    - 1D numpy array of indices in subimage_coords[0] of the pixel in image 1 that appears in the match in distances_matches, at the same position, therefore
    always equals to np.arange(y_lengths[0] * x_lengths[0])
    - 1D numpy array of indices in subimage_coords[1] of the pixel in image 2 that appears in the match in distances_matches, at the same position
    """

    # initialize null array of distances_matches
    nb_matches = y_lengths[0] * x_lengths[0]
    distances_matches = np.zeros((nb_matches,), dtype=np.float32)
    idx1_matches = np.zeros((nb_matches,), dtype=np.int32)
    idx2_matches = np.zeros((nb_matches,), dtype=np.int32)

    for idx_pixel_im1 in numba.prange(len(subimage_descriptors[0])):

        descrip_pixel_im1 = subimage_descriptors[0][idx_pixel_im1]

        # Initialize minimum distance to descriptor of pixel1 and index of corresponding pixel2
        min_dist = np.inf
        min_idx = None

        for idx_pixel_im2 in range(len(subimage_descriptors[1])):

            descrip_pixel_im2 = subimage_descriptors[1][idx_pixel_im2]

            # compute index of distance in the distances_matches array
            pix2_dist = desc.compute_descriptor_distance(
                descrip_pixel_im1, descrip_pixel_im2
            )
            # update minimum distance and index
            if pix2_dist < min_dist or min_idx is None:
                min_dist = pix2_dist
                min_idx = idx_pixel_im2

        # store minimum distance and index
        distances_matches[idx_pixel_im1] = min_dist

        # store coordinates
        idx1_matches[idx_pixel_im1] = idx_pixel_im1
        idx2_matches[idx_pixel_im1] = min_idx

    return distances_matches, idx1_matches, idx2_matches


def compute_and_save_distances(
    subimage_descriptors,
    y_lengths,
    x_lengths,
    y_starts,
    x_starts,
    photo_name,
    distance_type="min",
):
    before = datetime.now()
    print(f"Start computing distances: {before}")

    if distance_type == "all":
        distances_matches, idx_im1_matches, idx_im2_matches = (
            compute_distances_matches_pairs(subimage_descriptors, y_lengths, x_lengths)
        )
    elif distance_type == "min":
        distances_matches, idx_im1_matches, idx_im2_matches = (
            compute_minimal_distances_matches_pairs(
                subimage_descriptors, y_lengths, x_lengths
            )
        )
    else:
        raise ValueError("Invalid distance_type. Expected 'all' or 'min'.")

    after = datetime.now()
    print(f"End computing distances: {after}")
    print(f"Compute time: {after - before}")
    print(f"Chosen distance type: {distance_type}")

    # save distances and indices of pixels in the matches
    np.save(
        f"{dist_path}/{dist_filename}",
        distances_matches,
    )

    idx_ims_matches = [idx_im1_matches, idx_im2_matches]
    for id_image in range(2):
        np.save(
            f"{dist_path}/{matched_idx_filenames[id_image]}",
            idx_ims_matches[id_image],
        )


if __name__ == "__main__":

    # Load descriptors and list of pixel coordinates for both images

    # define number of images to load
    nb_images = 2
    # initiliaze empty lists of flat descriptors and coordinates
    subimage_descriptors = [None for i in range(nb_images)]
    subimage_coords = [None for i in range(nb_images)]

    # where to load descriptors

    for id_image in range(nb_images):

        subimage_descriptors[id_image] = np.load(
            f"{descrip_path}/{descrip_filenames[id_image]}.npy"
        )
        subimage_coords[id_image] = np.load(
            f"{descrip_path}/{kp_coords_filenames[id_image]}.npy"
        )

        # Look at shapes
        print(
            f"same number of descriptors and coordinates for image {id_image}: {subimage_descriptors[id_image].shape[0] == subimage_coords[id_image].shape[0]}"
        )
        print(
            f"also equal to the number of pixels: {subimage_descriptors[id_image].shape[0] == y_lengths[id_image] * x_lengths[id_image]}"
        )

    # choose between all distances or minimal distances

    compute_and_save_distances(
        subimage_descriptors,
        y_lengths,
        x_lengths,
        y_starts,
        x_starts,
        photo_name,
        distance_type=distance_type,
    )
