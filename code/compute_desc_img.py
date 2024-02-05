import cv2 as cv
import numpy as np
import os
import sys

import matplotlib.pyplot as plt

import descriptor as desc
import visu_hessian as vh


from datetime import datetime

from numba import njit
import numba


@njit(parallel=True)
def compute_desc_pixels(
    overall_features, y_start, y_length, x_start, x_length, border_size=1
):
    """
    Compute descriptors for a set of pixels in an image
    overall_features: overall features for the image, computed within a border
    return 2 lists: list of flattened descriptors, and list of pixels coordinates in the same order as the pixels
    pixel_position is (x, y)
    """
    img_descriptors = []
    coords_array = []
    # use numba.prange for parallelization
    for i in numba.prange(y_start, y_start + y_length):
        for j in range(x_start, x_start + x_length):
            # ensure kp_position is (horizontal=rows, vertical=cols)
            pixel_position = (j, i)

            descrip = desc.compute_descriptor_histograms_1_2_rotated(
                overall_features_1_2=overall_features,
                kp_position=pixel_position,
                nb_bins=1,
                bin_radius=2,
            )

            # flatten the list
            flat_descrip = desc.flatten_descriptor(descrip)

            img_descriptors.append(flat_descrip)
            coords_array.append(pixel_position)

    return img_descriptors, coords_array


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
# ):
#     """
#     Compute descriptors for a set of pixels in an image
#     overall_features: overall features for the image, computed within a border
#     return 2 lists: list of flattened descriptors, and list of pixels coordinates in the same order as the pixels
#     pixel_position is (x, y)
#     """
#     # compute length of descriptor
#     desc_length = 3 * nb_bins * nb_bins * (int(360 / delta_angle) + 1)
#     # initialize null arrays for pixel descriptors and coordinates
#     # img_descriptors = [[None for j in range(x_length)] for i in range(y_length)]
#     img_descriptors = np.zeros((y_length, x_length, desc_length), dtype=np.float32)
#     img_coords = np.zeros((y_length, x_length, 2), dtype=np.int32)
#     print(f"shape of img_descriptors: {img_descriptors.shape}")

#     # use numba.prange for parallelization
#     for i in numba.prange(y_start, y_start + y_length):
#         for j in range(x_start, x_start + x_length):
#             # ensure kp_position is (horizontal=rows, vertical=cols)
#             pixel_position = (j, i)

#             descrip = desc.compute_descriptor_histograms_1_2_rotated(
#                 overall_features_1_2=overall_features,
#                 kp_position=pixel_position,
#                 nb_bins=nb_bins,
#                 bin_radius=bin_radius,
#                 delta_angle=delta_angle,
#                 sigma=0,
#                 normalization_mode="global",
#             )

#             # flatten the list
#             flat_descrip = desc.flatten_descriptor(descrip)

#             img_descriptors[i - y_start, j - x_start] = flat_descrip
#             img_coords[i - y_start, j - x_start] = pixel_position

#     return img_descriptors, img_coords


if __name__ == "__main__":

    relative_path = "../data"
    img_folder = "blender/rocks"
    im_name1 = "rock_1_left"
    im_name2 = "rock_1_right"
    im_names = (im_name1, im_name2)
    im_ext = "png"

    ims = [
        cv.imread(
            f"{relative_path}/{img_folder}/{im_names[i]}.{im_ext}", cv.IMREAD_GRAYSCALE
        )
        for i in range(2)
    ]

    # plt.imshow(ims[0], cmap="gray")
    # plt.show()
    # plt.imshow(ims[1], cmap="gray")
    # plt.show()

    print("shapes of images", ims[0].shape, ims[1].shape)

    # compute float32 versions for calculations
    float_ims = [vh.convert_uint8_to_float32(ims[i]) for i in range(2)]

    # arbitrary sigma
    blur_sigma = 1.0
    float_ims = [desc.convolve_2D_gaussian(float_ims[i], blur_sigma) for i in range(2)]

    # compute descriptors for 2 images
    y_starts = [400, 400]
    y_lengths = [5, 5]
    x_starts = [800, 800]
    x_lengths = [5, 5]
    storage_folder = "computed_descriptors"

    for id_image in range(2):

        # compute for 2 image overall features
        before = datetime.now()
        print(f"feat computation beginning for image {id_image}:", before)
        border_size = 1
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
        img_descriptors, coords_array = compute_desc_pixels(
            overall_features,
            y_starts[id_image],
            y_lengths[id_image],
            x_starts[id_image],
            x_lengths[id_image],
            border_size,
        )
        after = datetime.now()
        print(f"desc computation end for image {id_image}", after)
        print(f"desc compute time for image {id_image}", after - before)

        # save img_descriptors and list of coordinates
        objects_to_save = [img_descriptors, coords_array]
        filename_prefix = f"{im_names[id_image]}_y_{y_starts[id_image]}_{y_lengths[id_image]}_x_{x_starts[id_image]}_{x_lengths[id_image]}"
        filename_suffixes = ["descs", "coords"]

        for id_object in range(2):
            np.save(
                f"{storage_folder}/{filename_prefix}_{filename_suffixes[id_object]}.npy",
                objects_to_save[id_object],
            )
