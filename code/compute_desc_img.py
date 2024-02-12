import cv2 as cv
import numpy as np
import os
import sys


import matplotlib.pyplot as plt

from datetime import datetime

from numba import njit
import numba

import descriptor as desc
import visu_hessian as vh

# Load hyperparameters for the computation pipeline
from computation_pipeline_hyper_params import *
from filenames_creation import *


@njit(parallel=True)
def compute_desc_pixels(
    overall_features,
    y_start,
    y_length,
    x_start,
    x_length,
    border_size=1,
    nb_bins=1,
    bin_radius=2,
    delta_angle=5.0,
    sigma=0,
    normalization_mode="global",
):
    """
    Compute descriptors for a set of pixels in an image
    overall_features: overall features for the image, computed within a border
    return numpy arrays:
    numpyarray of flattened descriptors, of shape (n, 3 * nb_bins * nb_bins * nb_angular_bins)
    numpy array of pixels coordinates in the same order as the pixels, of shape (n, 2)
    where n is the number of pixels = y_length * x_length
    pixel_position is (x, y)
    """
    # initialize the descriptors and coords array
    nb_angular_bins = int(360.0 / delta_angle) + 1
    n = y_length * x_length
    img_descriptors = np.zeros(
        (n, 3 * nb_bins * nb_bins * nb_angular_bins), dtype=np.float32
    )
    coords = np.zeros((n, 2), dtype=np.int32)
    # use numba.prange for parallelization
    for i in numba.prange(y_start, y_start + y_length):
        # for j in numba.prange(x_start, x_start + x_length):  # careful about prange
        for j in range(x_start, x_start + x_length):  # careful about prange
            # ensure kp_position is (horizontal=rows, vertical=cols)
            pixel_position = (j, i)

            descrip = desc.compute_descriptor_histograms_1_2_rotated(
                overall_features_1_2=overall_features,
                kp_position=pixel_position,
                nb_bins=nb_bins,
                bin_radius=bin_radius,
                delta_angle=delta_angle,
                sigma=sigma,
                normalization_mode=normalization_mode,
            )

            # flatten the list
            flat_descrip = desc.flatten_descriptor(descrip)
            arr_idx = (i - y_start) * x_length + (j - x_start)
            img_descriptors[arr_idx] = flat_descrip
            coords[arr_idx] = np.array(pixel_position)

    return img_descriptors, coords


if __name__ == "__main__":

    # Load preprocessed images as numpy arrays
    float_ims = np.load(f"{blurred_imgs_filename}.npy")

    # compute descriptors for 2 images
    for id_image in range(2):

        # compute for 2 image overall features
        before = datetime.now()
        print(f"feat computation beginning for image {id_image}:", before)
        overall_features = desc.compute_features_overall_abs(
            float_ims[id_image], border_size=border_size
        )
        after = datetime.now()
        print(f"feat computation end for image {id_image}", after)
        print(f"feat compute time for image {id_image}", after - before)
        print(
            f"shape of overall_features[0] of image {id_image}",
            overall_features[0].shape,
        )

        before = datetime.now()
        print(f"desc computation beginning for image {id_image}", before)
        # pass the defined h-parameters
        img_descriptors, coords = compute_desc_pixels(
            overall_features,
            y_starts[id_image],
            y_lengths[id_image],
            x_starts[id_image],
            x_lengths[id_image],
            border_size,
            nb_bins=nb_bins,
            bin_radius=bin_radius,
            delta_angle=delta_angle,
            sigma=sigma,
            normalization_mode=normalization_mode,
        )
        after = datetime.now()
        print(f"desc computation end for image {id_image}", after)
        print(f"desc compute time for image {id_image}", after - before)

        # save img_descriptors and list of coordinates

        np.save(
            f"{descrip_path}/{descrip_filenames[id_image]}",
            img_descriptors,
        )
        np.save(
            f"{descrip_path}/{kp_coords_filenames[id_image]}",
            coords,
        )
