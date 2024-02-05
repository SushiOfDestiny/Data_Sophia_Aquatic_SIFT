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

# Goal: load descriptors arrays, each for 1 image, flatten them and compute the distances between them
# first use bruteforce matching


def compute_distances_matches_pairs(
    subimage_descriptors, subimage_coords, y_lengths, x_lengths
):
    """
    Compute distances_matches between all pairs of descriptors from 2 images
    subimage_descriptors: list of 2 list of arrays of descriptors, each for 1 image, each containing as much element as pixels, each element
    of shape (3 * nb_bins * nb_bins * nb_angular_bins, )
    return:
    - 1D array of distances_matches, with distances_matches[id_pix1 * nb_pix2 + id_pix2] = distance between pixel
    of coords subimage_coords[0][id_pix1] of image 1 and pixel subimage_coords[2][id_pix2] of image 2
    - 1D array of indices in subimage_coords[0] of the pixel in image 1 that appears in the match in distances_matches, at the same position
    - 1D array of indices in subimage_coords[1] of the pixel in image 2 that appears in the match in distances_matches, at the same position
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
            idx2_matches[dist_idx] = idx_pixel_im1

    return distances_matches, idx1_matches, idx2_matches


if __name__ == "__main__":
    photo_name = "rock_1"
    im_names = ["rock_1_left", "rock_1_right"]
    y_starts = [400, 400]
    y_lengths = [5, 5]
    x_starts = [800, 800]
    x_lengths = [5, 5]

    # Load descriptors and list of pixel coordinates for both images

    # define number of images to load
    nb_images = 2
    # initilize empty lists of flat descriptors and coordinates
    subimage_descriptors = [None for i in range(nb_images)]
    subimage_coords = [None for i in range(nb_images)]

    # where to load descriptors
    storage_folder = "computed_descriptors"
    filename_suffixes = ["descs", "coords"]

    for id_image in range(nb_images):

        filename_prefix = f"{im_names[id_image]}_y_{y_starts[id_image]}_{y_lengths[id_image]}_x_{x_starts[id_image]}_{x_lengths[id_image]}"

        subimage_descriptors[id_image] = np.load(
            f"{storage_folder}/{filename_prefix}_{filename_suffixes[0]}.npy"
        )
        subimage_coords[id_image] = np.load(
            f"{storage_folder}/{filename_prefix}_{filename_suffixes[1]}.npy"
        )

        # Look at shapes
        print(
            f"same number of descriptors and coordinates for image {id_image}: {subimage_descriptors[id_image].shape[0] == subimage_coords[id_image].shape[0]}"
        )
        print(
            f"also equal to the number of pixels: {subimage_descriptors[id_image].shape[0] == y_lengths[id_image] * x_lengths[id_image]}"
        )

    # Remark: some descriptors are missing, especially at the border of the cropped image
    # they shouldn't

    # display them
    # for id_image in range(nb_images):
    #     # display first descriptor of the image
    #     for i in range(1):
    #         for j in range(1):
    #             visu_desc.display_descriptor(
    #                 subimage_descriptors[id_image][i * x_lengths[id_image] + j]
    #             )
    #     visu_desc.display_descriptor(subimage_descriptors[i])

    # # flatten them
    # flat_descriptors = [
    #     [
    #         desc.flatten_descriptor(subimage_descriptors[i][pixel_pos])
    #         for pixel_pos in range(y_lengths[i] * x_lengths[i])
    #     ]
    #     for i in range(nb_images)
    # ]

    # compute distances from descriptors

    before = datetime.now()
    nb_matches = y_lengths[0] * x_lengths[0] * y_lengths[1] * x_lengths[1]
    print(f"Start computing n={nb_matches} distances: {before}")

    # try to njit it, and to parallelize it, but encounters an issue with the
    # compute_descriptor_distance_unflat function
    distances_matches, idx_im1_matches, idx_im2_matches = (
        compute_distances_matches_pairs(
            subimage_descriptors, subimage_coords, y_lengths, x_lengths
        )
    )

    after = datetime.now()
    print(f"End computing n={nb_matches} distances: {after}")
    print(f"Compute time: {after - before}")

    # 6 250 000 distances computed in 1'20"

    # save distances and indices of pixels in the matches
    # where to save the distances
    target_folder = "computed_distances"
    target_filename_suffixes = ["dists", "matched_idx_im1", "matched_idx_im2"]
    objects_to_save = [distances_matches, idx_im1_matches, idx_im2_matches]
    target_filename_prefix = f"{photo_name}_y_{y_starts[0]}_{y_starts[1]}_{y_lengths[0]}_{y_lengths[1]}_{x_starts[0]}_{x_starts[1]}_{x_lengths[0]}_{x_lengths[1]}"

    for id_object in range(3):
        np.save(
            f"{target_folder}/{target_filename_prefix}_{target_filename_suffixes[id_object]}.npy",
            objects_to_save[id_object],
        )
