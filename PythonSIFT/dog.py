import sys, os

from pysift import (
    generateBaseImage,
    computeNumberOfOctaves,
    generateGaussianKernels,
    generateGaussianImages,
    generateDoGImages,
)
import matplotlib.pyplot as plt

from cv2 import imread, imshow, waitKey

import logging
import numpy as np


def visualize_DoG(
    image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5
):
    image = image.astype("float32")
    base_image = generateBaseImage(image, sigma, assumed_blur)
    num_octaves = computeNumberOfOctaves(base_image.shape)
    gaussian_kernels = generateGaussianKernels(sigma, num_intervals)
    gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels)
    dog_images = generateDoGImages(gaussian_images)
    fig, ax = plt.subplots(dog_images.shape[0], 1)
    vmin = np.min([np.min([[np.min(x) for x in m] for m in dog_images])])
    vmax = np.max([np.max([[np.max(x) for x in m] for m in dog_images])])
    for i in range(dog_images.shape[0]):
        dog_row = np.concatenate(dog_images[i, :], axis=1)
        im = ax[i].imshow(dog_row, vmin=vmin, vmax=vmax, cmap='gray')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    visualize_DoG(imread("box.png", 0))

    # # visualize for aqua picture
    # # load image from absolute path

    # script_dir = sys.path[0]
    # aqua_image_path = "../discover_opencv/images/aqua_images/"
    # image_name = "87_img_"
    # image_extension = ".png"
    # img_path = os.path.join(script_dir, aqua_image_path + image_name + image_extension)
    # # img_path = aqua_image_path + image_name + image_extension

    # # load image
    # imread(img_path, 0)
