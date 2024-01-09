import logging
import pysift

import cv2 as cv
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST

from numpy import (
    all,
    any,
    array,
    arctan2,
    cos,
    sin,
    exp,
    dot,
    log,
    logical_and,
    roll,
    sqrt,
    stack,
    trace,
    unravel_index,
    pi,
    deg2rad,
    rad2deg,
    where,
    zeros,
    floor,
    full,
    nan,
    isnan,
    round,
    float32,
)

from numpy.linalg import det, lstsq, norm


####################
# Global variables #
####################

logger = logging.getLogger(__name__)
float_tolerance = 1e-7


def compute_fitted_extrema(
    image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5
):
    """Compute SIFT extrema after quadratic fitting"""
    image = image.astype("float32")
    base_image = pysift.generateBaseImage(image, sigma, assumed_blur)
    num_octaves = pysift.computeNumberOfOctaves(base_image.shape)
    gaussian_kernels = pysift.generateGaussianKernels(sigma, num_intervals)
    gaussian_images = pysift.generateGaussianImages(
        base_image, num_octaves, gaussian_kernels
    )
    dog_images = pysift.generateDoGImages(gaussian_images)
    keypoints = pysift.findScaleSpaceExtrema(
        gaussian_images, dog_images, num_intervals, sigma, image_border_width
    )

    return keypoints


def find_scale_space_raw_extrema(
    gaussian_images,
    dog_images,
    num_intervals,
    sigma,
    image_border_width,
    contrast_threshold=0.04,
):
    """Find pixel positions of all scale-space extrema in the image pyramid"""
    logger.debug("Finding scale-space extrema...")
    threshold = floor(
        0.5 * contrast_threshold / num_intervals * 255
    )  # from OpenCV implementation
    extrema = []

    for octave_index, dog_images_in_octave in enumerate(dog_images):
        for image_index, (first_image, second_image, third_image) in enumerate(
            zip(
                dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:]
            )
        ):
            # (i, j) is the center of the 3x3 array
            for i in range(
                image_border_width, first_image.shape[0] - image_border_width
            ):
                for j in range(
                    image_border_width, first_image.shape[1] - image_border_width
                ):
                    if pysift.isPixelAnExtremum(
                        first_image[i - 1 : i + 2, j - 1 : j + 2],
                        second_image[i - 1 : i + 2, j - 1 : j + 2],
                        third_image[i - 1 : i + 2, j - 1 : j + 2],
                        threshold,
                    ):
                        # make sure the new pixel_cube will lie entirely within the image
                        # skipped step

                        # create new raw extremum as keypoint, without orientation
                        extremum = cv.KeyPoint()
                        extremum.pt = (
                            j * (2**octave_index),
                            i * (2**octave_index),
                        )
                        extremum.octave = octave_index + image_index * (2**8)
                        extremum.size = (
                            sigma
                            * (2 ** ((image_index) / float32(num_intervals)))
                            * (2 ** (octave_index + 1))
                        )  # octave_index + 1 because the input image was doubled

                        # add new extremum to the list of keypoints
                        extrema.append(extremum)

    return extrema


def compute_raw_extrema(
    image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5
):
    """Compute SIFT extrema before quadratic fitting"""
    image = image.astype("float32")
    base_image = pysift.generateBaseImage(image, sigma, assumed_blur)
    num_octaves = pysift.computeNumberOfOctaves(base_image.shape)
    gaussian_kernels = pysift.generateGaussianKernels(sigma, num_intervals)
    gaussian_images = pysift.generateGaussianImages(
        base_image, num_octaves, gaussian_kernels
    )
    dog_images = pysift.generateDoGImages(gaussian_images)
    keypoints = find_scale_space_raw_extrema(
        gaussian_images, dog_images, num_intervals, sigma, image_border_width
    )

    return keypoints


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    print("oui")
