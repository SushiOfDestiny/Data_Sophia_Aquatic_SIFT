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

if __name__ == "__main__":
    # Load images
    ims = [
        cv.imread(
            f"{original_imgs_path_prefix}/{im_names[i]}.{im_ext}", cv.IMREAD_GRAYSCALE
        )
        for i in range(2)
    ]
    print("shapes of images", ims[0].shape, ims[1].shape)

    # compute float32 versions for calculations
    float_ims = [vh.convert_uint8_to_float32(ims[i]) for i in range(2)]

    # blur images
    float_ims = [desc.convolve_2D_gaussian(float_ims[i], blur_sigma) for i in range(2)]

    # Save blurred images
    np.save(blurred_imgs_path, float_ims)

    print(f"Blurred images saved in {blurred_imgs_path}")

    # show cropped images
    cropped_float_ims = [
        float_ims[id_image][
            y_starts[id_image] : y_starts[id_image] + y_lengths[id_image],
            x_starts[id_image] : x_starts[id_image] + x_lengths[id_image],
        ]
        for id_image in range(2)
    ]

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    for id_image in range(2):
        axs[id_image].imshow(cropped_float_ims[id_image], cmap="gray")

    plt.show()
