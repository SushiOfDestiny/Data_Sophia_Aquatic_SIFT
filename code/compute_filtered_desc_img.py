# Goal of this script is to compute features of the images, filter points according to curvature, and compute their descriptors.

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


def compute_mean_abs_curv_arr(eigvals):
    """
    Compute the mean absolute curvature for each pixel in the image
    eigvals: numpy array of shape (n, m, 2) containing the eigenvalues of the Hessian matrix
    return: numpy array of shape (n, m) containing the mean absolute curvature for each pixel
    """
    # compute the mean absolute curvature for each pixel
    mean_abs_curv = np.zeros(eigvals.shape[:2], dtype=np.float32)
    for i in range(eigvals.shape[0]):
        for j in range(eigvals.shape[1]):
            # compute the mean absolute curvature
            mean_abs_curv[i, j] = 0.5 * (
                np.abs(eigvals[i, j, 0]) + np.abs(eigvals[i, j, 1])
            )
    return mean_abs_curv


def mask_percentile(arr, percentile=50, threshold=0.001):
    """
    Set to 0 the value of an array that are below a given percentile, and 1 to the other
    arr: numpy array
    percentile: percentile value
    threshold: threshold value
    return: numpy array of same shape with 0 at position of excluded pixels and 1 at position of accepted pixels
    """
    # compute the percentile value
    max_val = np.percentile(arr, percentile)
    # mask the pixels
    arr[arr < max_val] = 0
    mask_array = np.where(arr > threshold, 1, 0)
    return mask_array


@njit(parallel=True)
def filter_compute_desc_pixels(
    overall_features,
    y_start,
    y_length,
    x_start,
    x_length,
    mask_array,
    border_size=1,
    nb_bins=1,
    bin_radius=2,
    delta_angle=5.0,
    sigma=0,
    normalization_mode="global",
):
    """
    Compute descriptors for a set of pixels in an image
    overall_features: list of numpy arrays, of same shape (h, w) as whole image, overall features for the image, computed within a border
    mask_array: numpy array of same shape (h, w) as whole image containing the mask for the pixels, pixels of value 0 are not computed
    return numpy arrays:
    numpy array of flattened descriptors, of shape (n, 3 * nb_bins * nb_bins * nb_angular_bins)
    numpy array of pixels coordinates in the same order as the pixels, of shape (n, 2)
    where n is the number of filtered pixels (<= y_length * x_length)
    pixel_position is (x, y)
    """
    # initialize the descriptors and coords as numpy array
    nb_filtered_pixels = np.sum(np.where(mask_array > 0, 1, 0))
    nb_angular_bins = int(360.0 / delta_angle) + 1
    img_descriptors = np.zeros(
        shape=(nb_filtered_pixels, 3 * nb_bins * nb_bins * nb_angular_bins),
        dtype=np.float32,
    )
    coords = np.zeros(shape=(nb_filtered_pixels, 2), dtype=np.int32)

    # initialize index in arrays
    arr_idx = 0
    # use numba.prange for parallelization
    for i in numba.prange(y_start, y_start + y_length):
        # for j in numba.prange(x_start, x_start + x_length):  # careful about prange
        for j in range(x_start, x_start + x_length):  # careful about prange
            if mask_array[i, j] > 0:
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
                img_descriptors[arr_idx] = flat_descrip
                coords[arr_idx] = np.array(pixel_position)

                # update array index
                arr_idx += 1

    return img_descriptors, coords


if __name__ == "__main__":
    # Load preprocessed images as numpy arrays
    float_ims = np.load(f"{blurred_imgs_path}.npy")

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

        print("filter pixel by mean absolute curvature percentile in subimage")
        percentile = 50
        y_slice = slice(y_starts[id_image], y_starts[id_image] + y_lengths[id_image])
        x_slice = slice(x_starts[id_image], x_starts[id_image] + x_lengths[id_image])
        abs_eigvals = overall_features[1]
        mean_abs_curvs = compute_mean_abs_curv_arr(abs_eigvals)
        # mask the pixels with a mean absolute curvature below the chosen percentile
        # mask array is same shape as feature compute and therefore whole image
        masked_array = np.zeros(shape=(float_ims[id_image].shape[:2]), dtype=np.int32)
        # compute percentile only in subimage
        masked_array[y_slice, x_slice] = mask_percentile(
            mean_abs_curvs[y_slice, x_slice], percentile=percentile
        )

        # print some stats about mean curvatures and filtered points
        print(f"Min mean curvature: {np.min(mean_abs_curvs[y_slice, x_slice])}")
        print(f"Max mean curvature: {np.max(mean_abs_curvs[y_slice, x_slice])}")
        print(f"Mean mean curvature: {np.mean(mean_abs_curvs[y_slice, x_slice])}")
        print(
            f"Standard deviation of the mean curvatures: {np.std(mean_abs_curvs[y_slice, x_slice])}"
        )
        print(
            f"Percentile {percentile}%: {np.percentile(mean_abs_curvs[y_slice, x_slice], percentile)}"
        )

        print(
            f"number of pixels in subimage: {y_lengths[id_image] * x_lengths[id_image]}"
        )
        print(f"number of filtered accepted pixels: {np.sum(masked_array)}")
        print(
            f"percentage of filtered accepted pixels: {np.sum(masked_array) / y_lengths[id_image] * x_lengths[id_image]}"
        )

        before = datetime.now()
        print(f"desc computation beginning for image {id_image}", before)
        # compute descriptors and coords only for filtered pixels
        filtered_imgs_descs, filtered_kp_coords = filter_compute_desc_pixels(
            overall_features,
            y_starts[id_image],
            y_lengths[id_image],
            x_starts[id_image],
            x_lengths[id_image],
            mask_array=masked_array,
        )
        after = datetime.now()
        print(f"desc computation end for image {id_image}", after)
        print(f"desc compute time for image {id_image}", after - before)

        # save img_descriptors and list of coordinates, with special filt_suffix
        np.save(
            f"{descrip_path}/{descrip_filenames[id_image]}{filt_suffix}",
            filtered_imgs_descs,
        )
        np.save(
            f"{descrip_path}/{kp_coords_filenames[id_image]}{filt_suffix}",
            filtered_kp_coords,
        )
