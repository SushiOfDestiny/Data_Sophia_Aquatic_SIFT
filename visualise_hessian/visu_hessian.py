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


def visualize_curvature_values(g_img, keypoint, zoom_radius):
    """
    Compute eigenvalues of the Hessian matrix of all pixels in a zoomed area around a keypoint.
    Does nothing if the zoomed area is not in the image.
    g_img: grayscale image
    keypoint: SIFT keypoint
    zoom_radius: radius of the zoomed area in pixels
    Return: the matplotlib figure
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

        # # Normalize eigenvalues with the max absolute value
        max_abs_eigvals = np.max(np.abs(eigvals))
        normalized_eigvals = eigvals / max_abs_eigvals
        # normalized_eigvals = eigvals

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
        im1 = axs[1].imshow(eigvals1, cmap="gray", vmin=0, vmax=1)
        # im1 = axs[1].imshow(eigvals1, cmap="gray", vmin=-1, vmax=1)
        axs[1].set_title("eigenvalue 1")
        axs[1].axis("off")
        # add red pixel on the keypoint
        axs[1].scatter([zoom_radius], [zoom_radius], c="r")

        im2 = axs[2].imshow(eigvals2, cmap="gray", vmin=0, vmax=1)
        # im2 = axs[2].imshow(eigvals2, cmap="gray", vmin=-1, vmax=1)
        axs[2].set_title("eigenvalue 2")
        axs[2].axis("off")
        # add red pixel on the keypoint
        axs[2].scatter([zoom_radius], [zoom_radius], c="r")

        # # Define the same normalization for colorbars 1 and 2
        # norm = colors.Normalize(vmin=0, vmax=1)

        # Add colorbar and adjust its size so it matches the images
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

        # add legend
        fig.suptitle(f"SIFT Keypoint {y_kp}, {x_kp} (in red)", fontsize=10)

        # plt.tight_layout()
        # plt.show()

        return fig


def visualize_curvature_directions(g_img, keypoint, zoom_radius):
    """
    Compute eigenvectors of the Hessian matrix of all pixels in a zoomed area around a keypoint.
    display the directions in 2 colors depending on the sign of the eigenvalues.
    Does nothing if the zoomed area is not in the image.
    g_img: grayscale image
    keypoint: SIFT keypoint
    zoom_radius: radius of the zoomed area in pixels
    Return: the matplotlib figure
    """
    # compute pixel coordinates of the keypoint
    y, x = keypoint.pt
    y_kp = np.round(y).astype(int)
    x_kp = np.round(x).astype(int)

    # check if zoomed area is in the image
    is_in_image = (
        y_kp - zoom_radius >= 0
        and y_kp + zoom_radius < g_img.shape[0]
        and x_kp - zoom_radius >= 0
        and x_kp + zoom_radius < g_img.shape[1]
    )

    if not is_in_image:
        print("zoomed area is not fully in the image")
    else:
        # crop image around keypoint
        sub_img = g_img[
            y_kp - zoom_radius : y_kp + zoom_radius,
            x_kp - zoom_radius : x_kp + zoom_radius,
        ]
        # Compute hessian eigenvectors and eigenvalues of all pixels in subimage, (excluding a pixel border).
        border_size = 1
        h, w = sub_img.shape
        eigvects = np.zeros((h, w, 2, 2), dtype=np.float32)
        eigvals = np.zeros((h, w, 2), dtype=np.float32)

        # Downsample the computations by taking 1 pixel every step in each direction, instead of all pixels
        step = 5
        for y in range(border_size, h - border_size, step):
            for x in range(border_size, w - border_size, step):
                # assert y,x do not belong to keypoint, because it is calculated afterwards
                if y != zoom_radius or x != zoom_radius:
                    H = compute_hessian(sub_img, (y, x))
                    # simultaneously compute eigenvalues and eigenvectors
                    eigvals[y, x], eigvects[y, x] = np.linalg.eig(H)
        # also compute eigenvalues and eigenvectors for the keypoint
        H_kp = compute_hessian(sub_img, (zoom_radius, zoom_radius))
        (
            eigvals[zoom_radius, zoom_radius],
            eigvects[zoom_radius, zoom_radius],
        ) = np.linalg.eig(H_kp)

        # normalize eigenvectors with the max absolute value
        norms_eigvects = np.linalg.norm(eigvects, axis=-1)
        max_abs_eigvects = np.max(norms_eigvects)
        normalized_eigvects = eigvects / max_abs_eigvects

        # plot subimage and eigenvectors
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # plot subimage
        axs[0].imshow(sub_img, cmap="gray")
        axs[0].set_title(f"zoomed image, radius={zoom_radius}")
        axs[0].axis("off")
        # add red pixel on the keypoint
        kp_factor = 0.1
        axs[0].scatter([zoom_radius], [zoom_radius], c="r", s=zoom_radius * kp_factor)

        # plot eigenvectors
        # define colormap for eigenvectors depending on the value of the eigenvalues
        # the higher the eigenvalue, the more red the eigenvector, the lower the eigenvalue, the more blue the eigenvector
        colormap = plt.cm.get_cmap("RdBu")
        # normalize eigenvalues with the min and max eigenvalues
        vmin, vmax = np.min(eigvals), np.max(eigvals)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        # plot eigenvectors on another image
        axs[1].imshow(sub_img, cmap="gray")

        for y in range(border_size, h - border_size, step):
            for x in range(border_size, w - border_size, step):
                # assert y,x do not belong to keypoint, because it is calculated afterwards
                if y != zoom_radius or x != zoom_radius:
                    # compute color of the eigenvector depending on the eigenvalues
                    color1 = colormap(norm(eigvals[y, x, 0]))
                    # plot eigenvector 1
                    axs[1].arrow(
                        x,
                        y,
                        normalized_eigvects[y, x, 0, 0],
                        normalized_eigvects[y, x, 0, 1],
                        color=color1,
                        width=0.1,
                    )
                    # compute color of the eigenvector depending on the eigenvalues
                    color2 = colormap(norm(eigvals[y, x, 1]))
                    # plot eigenvector 2
                    axs[1].arrow(
                        x,
                        y,
                        normalized_eigvects[y, x, 1, 0],
                        normalized_eigvects[y, x, 1, 1],
                        color=color2,
                        width=0.1,
                    )
        # display eigenvectors of the keypoint
        # plot eigenvector 1
        color_kp1 = colormap(norm(eigvals[zoom_radius, zoom_radius, 0]))
        axs[1].arrow(
            zoom_radius,
            zoom_radius,
            normalized_eigvects[zoom_radius, zoom_radius, 0, 0],
            normalized_eigvects[zoom_radius, zoom_radius, 0, 1],
            color=color_kp1,
            width=0.1,
        )
        # plot eigenvector 2
        color_kp2 = colormap(norm(eigvals[zoom_radius, zoom_radius, 1]))
        axs[1].arrow(
            zoom_radius,
            zoom_radius,
            normalized_eigvects[zoom_radius, zoom_radius, 1, 0],
            normalized_eigvects[zoom_radius, zoom_radius, 1, 1],
            color=color_kp2,
            width=0.1,
        )

        # add red pixel on the keypoint, with variable size
        axs[1].scatter(
            [zoom_radius],
            [zoom_radius],
            c="r",
            s=zoom_radius * kp_factor,
        )
        axs[1].set_title("eigenvectors")
        axs[1].axis("off")

        # add the blue to red colormap of the arrows
        # create a ScalarMappable with the same colormap and normalization as the arrows
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])

        # add the colorbar of the colormap of the arrows
        fig.colorbar(sm, ax=axs[1], fraction=0.046, pad=0.04)

        # add legend
        fig.suptitle(f"SIFT Keypoint {y_kp}, {x_kp} (in red)", fontsize=10)

        return fig


# TESTS
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    print("oui")

    # load grayscale image
    im_name = "87_img_"
    img = cv.imread(f"../images/{im_name}.png", 0)
    # # show image
    # # plt.imshow(img, cmap="gray")
    # # plt.show()

    # define folder to save images
    img_folder = "zoomed_kp"

    # calculate sift keypoints and descriptors
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    # draw colormap of eigenvalues of Hessian matrix for 1 keypoint
    kp0 = keypoints[550]
    y_kp0, x_kp0 = np.round(kp0.pt).astype(int)
    # print(kp0.pt)

    #####################
    # Test visualize_curvature_values #

    eigval_fig = visualize_curvature_values(img, kp0, 30)

    plt.figure(eigval_fig.number)
    plt.show()

    # # save figure
    # eigval_fig.savefig(
    #     f"zoomed_kp/zoomed_{im_name}_kp_{y_kp}_{x_kp}_{zoom_radius}.png",
    #     dpi="figure",
    # )

    # draw colormap of eigenvalues of Hessian matrix for some keypoints
    # zoom_radius = 10
    # for kp in keypoints[50:301:50]:
    #     visualize_curvature_values(im_name, img, kp, zoom_radius)

    #######################################
    # Test visualize_curvature_directions #
    zoom_radius = 30
    eig_fig = visualize_curvature_directions(img, kp0, zoom_radius)

    plt.figure(eig_fig.number)
    plt.show()

    # # save figure
    # eig_fig.savefig(
    #     f"{img_folder}/zoomed_{im_name}_kp_{y_kp0}_{x_kp0}_{zoom_radius}_eigvects.png",
    #     dpi="figure",
    # )
