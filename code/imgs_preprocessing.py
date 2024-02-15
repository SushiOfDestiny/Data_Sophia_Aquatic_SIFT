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

from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, black_tophat, white_tophat, disk

from skimage import exposure, img_as_ubyte


if __name__ == "__main__":

    # Load images
    ims = [
        cv.imread(
            f"{original_imgs_path_prefix}/{im_names[i]}.{im_ext}", cv.IMREAD_GRAYSCALE
        )
        for i in range(2)
    ]
    print("shapes of images", ims[0].shape, ims[1].shape)

    # ensure that the images are uint8
    ims = [img_as_ubyte(image) for image in ims]

    # Apply histogram equalization
    if adapt_hist_eq is not None and adapt_hist_eq == True:
        ims = [exposure.equalize_adapthist(image, clip_limit=0.01) for image in ims]

    # Apply threshold and binary erosion
    # Calculate Otsu's threshold
    # lower_thresholds = [1.1 * threshold_otsu(images) for images in ims]
    # # Apply the lower threshold to create a binary image
    # ims = [ims[id_image] > lower_thresholds[id_image] for id_image in range(2)]
    # ims = [binary_erosion(ims[id_image]) for id_image in range(2)]
    # # Apply white top-hat transformation
    # ims = [white_tophat(ims[id_image], disk(5)) for id_image in range(2)]
    # # Apply black top-hat transformation
    # ims = [black_tophat(ims[id_image], disk(5)) for id_image in range(2)]

    # fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    # for id_image in range(2):
    #     axs[id_image].imshow(ims[id_image], cmap="gray")

    # plt.show()

    # compute float32 versions for calculations
    float_ims = [vh.convert_uint8_to_float32(ims[i]) for i in range(2)]

    # blur images, unless blur_sigma is 0.
    if blur_sigma > 0.1:
        float_ims = [
            desc.convolve_2D_gaussian(float_ims[i], blur_sigma) for i in range(2)
        ]

    # Save blurred images
    np.save(blurred_imgs_filename, float_ims)

    print(f"Blurred images saved in {original_imgs_path_prefix}")

    # show cropped images
    cropped_ims = [
        ims[id_image][
            y_starts[id_image] : y_starts[id_image] + y_lengths[id_image],
            x_starts[id_image] : x_starts[id_image] + x_lengths[id_image],
        ]
        for id_image in range(2)
    ]

    cropped_float_ims = [
        float_ims[id_image][
            y_starts[id_image] : y_starts[id_image] + y_lengths[id_image],
            x_starts[id_image] : x_starts[id_image] + x_lengths[id_image],
        ]
        for id_image in range(2)
    ]

    # display cropped float images
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    for id_image in range(2):
        axs[id_image].imshow(cropped_float_ims[id_image], cmap="gray")

    plt.show()

    # save cropped  int images for sift
    for id_image in range(2):
        cv.imwrite(
            f"{original_imgs_path_prefix}/{cropped_ims_filenames[id_image]}.{im_ext}",
            cropped_ims[id_image],
        )
    # save cropped  int images for sift
    for id_image in range(2):
        np.save(
            f"{original_imgs_path_prefix}/{cropped_float_ims_filenames[id_image]}",
            cropped_float_ims[id_image],
        )
    print(f"Cropped images saved in {original_imgs_path_prefix}")
