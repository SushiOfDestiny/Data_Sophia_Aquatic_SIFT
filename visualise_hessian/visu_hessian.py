import os
import sys
import logging

sys.path.append("../PythonSIFT")
import pysift

# return to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

import cv2 as cv

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


def compute_gradient(g_img, position):
    """
    Compute gradient with finite differences at order 1.
    g_img: grayscale image
    position: (y,x) position of point in pixels
    """
    # force cast into float32 to avoid overflow
    g_img = g_img.astype(np.float32)

    y, x = position
    dx = g_img[y, x + 1] - g_img[y, x - 1]
    dy = g_img[y + 1, x] - g_img[y - 1, x]
    return np.array([dx, dy])


def compute_hessian(g_img, position):
    """
    Compute hessian with finite differences at order 2.
    g_img: grayscale image
    position: (y,x) position of the pixel
    """
    # force cast into float32 to avoid overflow
    g_img = g_img.astype(np.float32)

    y, x = position
    # compute derivatives
    dxx = g_img[y, x + 1] + g_img[y, x - 1] - 2 * g_img[y, x]
    dyy = g_img[y + 1, x] + g_img[y - 1, x] - 2 * g_img[y, x]
    dxy = (
        g_img[y + 1, x + 1]
        + g_img[y - 1, x - 1]
        - g_img[y + 1, x - 1]
        - g_img[y - 1, x + 1]
    ) / 4
    return np.array([[dxx, dxy], [dxy, dyy]])


def visu_hessian(im_name, g_img, keypoint, zoom_radius):
    """
    Compute eigenvalues of the Hessian matrix of all pixels in a zoomed area around a keypoint.
    Does nothing if the zoomed area is not in the image.
    g_img: grayscale image
    keypoint: SIFT keypoint
    zoom_radius: radius of the zoomed area in pixels
    """
    # compute pixel coordinates of the keypoint
    y, x = keypoint.pt
    y_kp = np.round(y).astype(int)
    x_kp = np.round(x).astype(int)
    print(y, x)

    # check if zoomed area is in the image
    is_in_image = (
        y_kp - zoom_radius >= 0
        and y_kp + zoom_radius < g_img.shape[0]
        and x_kp - zoom_radius >= 0
        and x_kp + zoom_radius < g_img.shape[1]
    )

    if is_in_image:
        # crop image around keypoint
        sub_img = g_img[
            y_kp - zoom_radius : y_kp + zoom_radius,
            x_kp - zoom_radius : x_kp + zoom_radius,
        ]
        # Compute hessian eigenvalues of all pixels in subimage, (excluding a pixel border).
        border_size = 1
        h, w = sub_img.shape
        eigvals = np.zeros((h, w, 2), dtype=np.float32)
        for y in range(border_size, h - border_size):
            for x in range(border_size, w - border_size):
                H = compute_hessian(sub_img, (y, x))
                eigvals[y, x] = np.linalg.eigvals(H)
        # Normalize eigenvalues with the max absolute value
        max_abs_eigvals = np.max(np.abs(eigvals), axis=(0, 1))
        normalized_eigvals = eigvals / max_abs_eigvals

        # Compute colormap images
        eigvals1 = normalized_eigvals[:, :, 0]
        eigvals2 = normalized_eigvals[:, :, 1]

        # Affine transform eigenvalues from [-1,1] to [0, 1]
        eigvals1 = (eigvals1 + 1) / 2
        eigvals2 = (eigvals2 + 1) / 2

        # Plot subimage and eigenvalues
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot subimage
        im0 = axs[0].imshow(sub_img, cmap="gray")
        axs[0].set_title(f"zoomed image, radius={zoom_radius}")
        axs[0].axis("off")
        # add red pixel on the keypoint
        axs[0].scatter([zoom_radius], [zoom_radius], c="r")

        # Plot colormap images
        im1 = axs[1].imshow(eigvals1, cmap="gray")
        axs[1].set_title("eigenvalue 1")
        axs[1].axis("off")
        # add red pixel on the keypoint
        axs[1].scatter([zoom_radius], [zoom_radius], c="r")

        im2 = axs[2].imshow(eigvals2, cmap="gray")
        axs[2].set_title("eigenvalue 2")
        axs[2].axis("off")
        # add red pixel on the keypoint
        axs[2].scatter([zoom_radius], [zoom_radius], c="r")

        # Define the same normalization for colorbars 1 and 2
        norm = colors.Normalize(vmin=0, vmax=1)

        # Add colorbar and adjust its size so it matches the images
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, norm=norm)
        fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04, norm=norm)

        # add legend
        fig.suptitle(f"SIFT Keypoint {y_kp}, {x_kp} (in red)", fontsize=10)

        plt.tight_layout()
        # plt.show()

        # save figure
        fig.savefig(f"zoomed_kp/zoomed_{im_name}_kp_{y_kp}_{x_kp}.png", dpi="figure")


# TESTS
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    print("oui")

    # load image
    # load grayscale image
    im_name = "87_img_"
    img = cv.imread(f"../images/{im_name}.png", 0)
    # show image
    # plt.imshow(img, cmap="gray")
    plt.show()

    # calculate sift keypoints and descriptors
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    # draw colormap of eigenvalues of Hessian matrix for 1 keypoint
    kp0 = keypoints[600]
    # print(kp0.pt)

    visu_hessian(im_name, img, kp0, 15)

    # draw colormap of eigenvalues of Hessian matrix for some keypoints
    zoom_radius = 10
    for kp in keypoints[50:301:50]:
        visu_hessian(im_name, img, kp, zoom_radius)
