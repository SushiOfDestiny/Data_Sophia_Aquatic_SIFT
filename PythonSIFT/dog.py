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


def visualize_DoG(
    image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5
):
    imshow("image", image)
    k = waitKey()
    image = image.astype("float32")
    base_image = generateBaseImage(image, sigma, assumed_blur)
    num_octaves = computeNumberOfOctaves(base_image.shape)
    gaussian_kernels = generateGaussianKernels(sigma, num_intervals)
    gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels)
    dog_images = generateDoGImages(gaussian_images)
    fig, ax = plt.subplots(dog_images.shape[0], dog_images.shape[1])
    for i in range(dog_images.shape[0]):
        for j in range(dog_images.shape[1]):
            ax[i, j].imshow(dog_images[i, j])
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
