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


def compute_non_null_coords(mask_array):
    """
    Compute the indices of non null pixels in a mask array
    mask_array: numpy array of shape (h, w) containing the mask for the pixels
    return: numpy array of shape (n, 2) containing the coordinates of non null pixels
    coordinates are (x, y)
    """
    coords = np.array([np.array([j, i]) for (i, j) in np.argwhere(mask_array > 0)])
    return coords


# @njit(parallel=True)
# def compute_desc_pixels(
#     overall_features,
#     y_start,
#     y_length,
#     x_start,
#     x_length,
#     border_size=1,
#     nb_bins=1,
#     bin_radius=2,
#     delta_angle=5.0,
#     sigma=0,
#     normalization_mode="global",
# ):
#     """
#     Compute descriptors for a set of pixels in an image
#     overall_features: overall features for the image, computed within a border
#     return numpy arrays:
#     numpyarray of flattened descriptors, of shape (n, 3 * nb_bins * nb_bins * nb_angular_bins)
#     numpy array of pixels coordinates in the same order as the pixels, of shape (n, 2)
#     where n is the number of pixels = y_length * x_length
#     pixel_position is (x, y)
#     """
#     # initialize the descriptors and coords array
#     nb_angular_bins = int(360.0 / delta_angle) + 1
#     n = y_length * x_length
#     imgs_descs = np.zeros(
#         (n, 3 * nb_bins * nb_bins * nb_angular_bins), dtype=np.float32
#     )
#     coords = np.zeros((n, 2), dtype=np.int32)
#     # use numba.prange for parallelization
#     for i in numba.prange(y_start, y_start + y_length):
#         # for j in numba.prange(x_start, x_start + x_length):  # careful about prange
#         for j in range(x_start, x_start + x_length):  # careful about prange
#             # ensure kp_position is (horizontal=rows, vertical=cols)
#             pixel_position = (j, i)

#             descrip = desc.compute_descriptor_histograms_1_2_rotated(
#                 overall_features_1_2=overall_features,
#                 kp_position=pixel_position,
#                 nb_bins=nb_bins,
#                 bin_radius=bin_radius,
#                 delta_angle=delta_angle,
#                 sigma=sigma,
#                 normalization_mode=normalization_mode,
#             )

#             # flatten the list
#             flat_descrip = desc.flatten_descriptor(descrip)
#             arr_idx = (i - y_start) * x_length + (j - x_start)
#             imgs_descs[arr_idx] = flat_descrip
#             coords[arr_idx] = np.array(pixel_position)

#     return imgs_descs, coords


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


def filter_by_mean_abs_curv(
    float_im, abs_eigvals, y_start, y_length, x_start, x_length, percentile
):
    """
    Filter the pixels of an image by a given percentile of mean absolute curvature
    float_im: numpy array of shape (h, w) containing the image
    abs_eigvals: numpy array of shape (h, w) containing the absolute value of eigenvalue for each pixel
    y_start, x_start: int, the start of the subimage
    percentile: int, the percentile to use for filtering
    return: numpy array of shape (h, w) containing the mask for the pixels
    return also the mean absolute curvature array, and the slice of the subimage
    """
    y_slice = slice(y_start, y_start + y_length)
    x_slice = slice(x_start, x_start + x_length)
    mean_abs_curvs = compute_mean_abs_curv_arr(abs_eigvals)
    # mask the pixels with a mean absolute curvature below the chosen percentile
    # mask array is same shape as feature compute and therefore whole image
    mask_array = np.zeros(shape=(float_im.shape[:2]), dtype=np.int32)
    # compute percentile only in subimage
    mask_array[y_slice, x_slice] = mask_percentile(
        mean_abs_curvs[y_slice, x_slice], percentile=percentile
    )

    return mask_array, mean_abs_curvs, y_slice, x_slice

def filter_by_std_neighbor_curv(
        float_im, eigvals, y_start, y_length, x_start, x_length, percentile, bin_radius
):
    """
    Filter the pixels of an image by a given percentile of standard deviation absolute curvature in a neighborhood
    
    Arguments:
    float_im: numpy array of shape (h, w) containing the image
    eigvals: numpy array of shape (h, w) containing the eigenvalues (curvature values) for each pixel
    y_start, x_start: int, the start of the subimage
    percentile: int, the percentile to use for filtering

    Returns:
    - mask_array: numpy array of shape (h, w) containing the filtered pixel mask    
    """

    y_slice = slice(y_start, y_start + y_length)
    x_slice = slice(x_start, x_start + x_length)

    eigval_means = np.mean(eigvals, axis=2) # compute mean of curvature values for each pixel

    bins_std = np.zeros(shape=(y_length, x_length), dtype=np.float32)
    for y in range(bin_radius, bins_std.shape[0]-bin_radius):
        for x in range(bin_radius, bins_std.shape[1]-bin_radius):
            bins_std[y, x] = np.std(eigval_means[y-bin_radius:y+bin_radius, x-bin_radius, x+bin_radius])
    
    # Compute percentile prefiltering mask
    mask = np.zeros(shape=float_im.shape[:2], dtype=bool)
    threshold = np.percentile(bins_std, percentile)
    mask[y_slice, x_slice] = bins_std > threshold
    
    return mask


# @njit(parallel=True)
# def filter_compute_desc_pixels(
#     overall_features,
#     y_start,
#     y_length,
#     x_start,
#     x_length,
#     mask_array,
#     border_size=1,
#     nb_bins=1,
#     bin_radius=2,
#     delta_angle=5.0,
#     sigma=0,
#     normalization_mode="global",
# ):
#     """
#     Compute descriptors for a set of pixels in an image
#     overall_features: list of numpy arrays, of same shape (h, w) as whole image, overall features for the image, computed within a border
#     mask_array: numpy array of same shape (h, w) as whole image containing the mask for the pixels, pixels of value 0 are not computed
#     return numpy arrays:
#     numpy array of flattened descriptors, of shape (n, 3 * nb_bins * nb_bins * nb_angular_bins)
#     numpy array of pixels coordinates in the same order as the pixels, of shape (n, 2), in the frame of the whole image
#     where n is the number of filtered pixels (<= y_length * x_length)
#     pixel_position is (x, y)
#     """
#     # initialize the descriptors and coords as numpy array
#     nb_filtered_pixels = np.sum(np.where(mask_array > 0, 1, 0))

#     nb_angular_bins = int(360.0 / delta_angle) + 1
#     img_descriptors = np.zeros(
#         shape=(nb_filtered_pixels, 3 * nb_bins * nb_bins * nb_angular_bins),
#         dtype=np.float32,
#     )
#     coords = np.zeros(shape=(nb_filtered_pixels, 2), dtype=np.int32)

#     # initialize index in arrays
#     arr_idx = 0
#     # use numba.prange for parallelization
#     # for i in numba.prange(y_start, y_start + y_length):
#     for i in numba.prange(y_start, y_start + y_length):
#         # for j in numba.prange(x_start, x_start + x_length):  # careful about prange
#         for j in range(x_start, x_start + x_length):  # careful about prange
#             if mask_array[i, j] > 0:
#                 # ensure kp_position is (horizontal=rows, vertical=cols)
#                 pixel_position = (j, i)

#                 descrip = desc.compute_descriptor_histograms_1_2_rotated(
#                     overall_features_1_2=overall_features,
#                     kp_position=pixel_position,
#                     nb_bins=nb_bins,
#                     bin_radius=bin_radius,
#                     delta_angle=delta_angle,
#                     sigma=sigma,
#                     normalization_mode=normalization_mode,
#                 )

#                 # flatten the list
#                 flat_descrip = desc.flatten_descriptor(descrip)
#                 img_descriptors[arr_idx] = flat_descrip
#                 coords[arr_idx] = np.array(pixel_position)

#                 # update array index
#                 arr_idx += 1

#     print(np.sum(coords[:, 0] == 0))

#     return img_descriptors, coords


@njit(parallel=True)
def compute_desc_pixels(
    overall_features,
    coords,
    border_size=1,
    nb_bins=1,
    bin_radius=2,
    delta_angle=5.0,
    sigma=0,
    normalization_mode="global",
):
    """
    Compute descriptors for a set of pixels coordinates in an image
    overall_features: list of numpy arrays, of same shape (h, w) as whole image, overall features for the image, computed within a border
    coords: np.array of coordinates of pixels to compute descriptor for (in the whole image), shape (n, 2),
    pixel_position is (x, y)
    return numpy arrays:
    numpy array of flattened descriptors, of shape (n, 3 * nb_bins * nb_bins * nb_angular_bins)
    """
    # initialize the descriptors and coords as numpy array
    nb_filtered_pixels = len(coords)

    nb_angular_bins = int(360.0 / delta_angle) + 1
    imgs_descs = np.zeros(
        shape=(nb_filtered_pixels, 3 * nb_bins * nb_bins * nb_angular_bins),
        dtype=np.float32,
    )

    for pix_idx in numba.prange(nb_filtered_pixels):
        pixel_position = coords[pix_idx]
        # ensure kp_position is (horizontal=rows, vertical=cols)

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
        imgs_descs[pix_idx] = flat_descrip

    return imgs_descs


def print_info_curvatures(
    mask_array, mean_abs_curvs, y_slice, x_slice, y_length, x_length, percentile
):
    """
    print some stats about mean curvatures and filtered points
    mask_array: numpy array of shape (h, w) containing the mask for the pixels
    mean_abs_curvs: numpy array of shape (h, w) containing the mean absolute curvature for each pixel
    y_slices, x_slices: slice objects to define the subimage (ban also be lists)
    y_length, x_length: int, the length of the subimage
    percentile: int, the percentile to use for filtering
    """
    print(f"Min mean curvature: {np.min(mean_abs_curvs[y_slice, x_slice])}")
    print(f"Max mean curvature: {np.max(mean_abs_curvs[y_slice, x_slice])}")
    print(f"Mean mean curvature: {np.mean(mean_abs_curvs[y_slice, x_slice])}")
    print(
        f"Standard deviation of the mean curvatures: {np.std(mean_abs_curvs[y_slice, x_slice])}"
    )
    print(
        f"Percentile {percentile}%: {np.percentile(mean_abs_curvs[y_slice, x_slice], percentile)}"
    )

    print(f"number of pixels in subimage: {y_length * x_length}")
    print(f"number of filtered accepted pixels: {np.sum(mask_array)}")
    print(
        f"percentage of filtered accepted pixels: {100. * np.sum(mask_array) / (y_length * x_length)}"
    )


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

        # switch function call depending on use_filt
        if not use_filt:

            before = datetime.now()
            print(f"desc computation beginning for image {id_image}", before)

            # create mask with ones only in subimage
            mask_array = np.zeros(shape=(float_ims[id_image].shape[:2]), dtype=np.int32)
            mask_array[
                y_starts[id_image] : y_starts[id_image] + y_lengths[id_image],
                x_starts[id_image] : x_starts[id_image] + x_lengths[id_image],
            ] = 1

        else:
            print("filter pixel by mean absolute curvature percentile in subimage")
            print(
                f"prefiltering by mean absolute curvature beginning for image {id_image}",
                before,
            )
            before = datetime.now()

            mask_array, mean_abs_curvs, y_slice, x_slice = filter_by_mean_abs_curv(
                float_ims[id_image],
                overall_features[1],
                y_starts[id_image],
                y_lengths[id_image],
                x_starts[id_image],
                x_lengths[id_image],
                percentile,
            )

            after = datetime.now()
            print(
                f"prefiltering by mean absolute curvature end for image {id_image}",
                after,
            )
            print(
                f"prefiltering by mean absolute curvature compute time for image {id_image}",
                after - before,
            )

            print_info_curvatures(
                mask_array,
                mean_abs_curvs,
                y_slice,
                x_slice,
                y_lengths[id_image],
                x_lengths[id_image],
                percentile,
            )

        # compute coordinates of prefiltered pixels
        kp_coords = compute_non_null_coords(mask_array)
        print(f"coords array shape: {kp_coords.shape}")

        before = datetime.now()
        print(f"desc computation beginning for image {id_image}", before)

        imgs_descs = compute_desc_pixels(
            overall_features,
            kp_coords,
            border_size=border_size,
            nb_bins=nb_bins,
            bin_radius=bin_radius,
            delta_angle=delta_angle,
            sigma=sigma,
            normalization_mode=normalization_mode,
        )

        after = datetime.now()
        print(f"desc computation end for image {id_image}", after)
        print(f"desc compute time for image {id_image}", after - before)

        # print number of n such as imgs_descs[n] = 0.
        print(
            f"Number of empty filtered descriptors {np.sum([1 for descrip in imgs_descs if np.sum(descrip) == 0])}"
        )

        # # plot the filtered pixels on each subimage side by side
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # for id_image in range(2):
        #     ax[id_image].imshow(float_ims[id_image], cmap="gray")
        #     ax[id_image].scatter(
        #         filtered_kp_coords[:, 0],
        #         filtered_kp_coords[:, 1],
        #         c="r",
        #         s=2,
        #     )
        #     ax[id_image].set_title(f"Filtered pixels in subimage {id_image}")
        # plt.show()

        # save imgs_descs and list of coordinates
        np.save(
            f"{descrip_path}/{descrip_filenames[id_image]}",
            imgs_descs,
        )
        np.save(
            f"{descrip_path}/{kp_coords_filenames[id_image]}",
            kp_coords,
        )
