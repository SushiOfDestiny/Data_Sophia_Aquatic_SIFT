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


def add_vector_to_ax(colormap, norm, cm_value, x, y, vect, ax, width=0.1):
    """
    Add a vector to an axis with a colormap depending on a value.
    colormap: matplotlib colormap
    norm: matplotlib norm
    cm_value: value to get the color from the colormap
    x, y: position of the vector
    vect: vector to plot
    ax: matplotlib axis
    width: width of the vector
    """
    # compute color of the eigenvector depending on the eigenvalue
    color = colormap(norm(cm_value))
    # plot eigenvector
    ax.arrow(x, y, vect[0], vect[1], color=color, width=width)


def visualize_curvature_directions(g_img, keypoint, zoom_radius, step_percentage=5):
    """
    Compute eigenvectors of the Hessian matrix of all pixels in a zoomed area around a keypoint.
    display the directions in 2 colors depending on the sign of the eigenvalues.
    Does nothing if the zoomed area is not in the image.
    g_img: grayscale image
    keypoint: SIFT keypoint
    zoom_radius: radius of the zoomed area in pixels
    step_percentage: percentage of pixels to skip in the computation of the eigenvectors, w.r.t subimage size
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
        step = (2 * zoom_radius + 1) * step_percentage / 100
        step = np.round(step).astype(int)

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
                    add_vector_to_ax(
                        colormap,
                        norm,
                        eigvals[y, x, 0],
                        x,
                        y,
                        normalized_eigvects[y, x, 0],
                        axs[1],
                    )
                    add_vector_to_ax(
                        colormap,
                        norm,
                        eigvals[y, x, 1],
                        x,
                        y,
                        normalized_eigvects[y, x, 1],
                        axs[1],
                    )
        # display eigenvectors of the keypoint
        add_vector_to_ax(
            colormap,
            norm,
            eigvals[zoom_radius, zoom_radius, 0],
            zoom_radius,
            zoom_radius,
            normalized_eigvects[zoom_radius, zoom_radius, 0],
            axs[1],
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


def visualize_curvature_directions_ax_sm(
    g_img, keypoint, zoom_radius, ax, step_percentage=5
):
    """
    g_img: grayscale image
    keypoint: SIFT keypoint
    zoom_radius: radius of the zoomed area in pixels
    step_percentage: percentage of pixels to skip in the computation of the eigenvectors, w.r.t subimage size
    ax: matplotlib axis

    Compute eigenvectors of the Hessian matrix of all pixels in a zoomed area around a keypoint.
    display the directions in 2 colors depending on the sign of the eigenvalues.
    Does nothing if the zoomed area is not in the image.
    Inplace modify the argument ax

    Return: the scalable colormap of the arrows
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
        step = (2 * zoom_radius + 1) * step_percentage / 100
        step = np.round(step).astype(int)

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

        # plot eigenvectors
        # define colormap for eigenvectors depending on the value of the eigenvalues
        # the higher the eigenvalue, the more red the eigenvector, the lower the eigenvalue, the more blue the eigenvector
        colormap = plt.cm.get_cmap("RdBu")

        # normalize eigenvalues with the min and max eigenvalues
        vmin, vmax = np.min(eigvals), np.max(eigvals)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        # plot eigenvectors on another image
        ax.imshow(sub_img, cmap="gray")

        for y in range(border_size, h - border_size, step):
            for x in range(border_size, w - border_size, step):
                # assert y,x do not belong to keypoint, because it is calculated afterwards
                if y != zoom_radius or x != zoom_radius:
                    add_vector_to_ax(
                        colormap,
                        norm,
                        eigvals[y, x, 0],
                        x,
                        y,
                        normalized_eigvects[y, x, 0],
                        ax,
                    )
                    add_vector_to_ax(
                        colormap,
                        norm,
                        eigvals[y, x, 1],
                        x,
                        y,
                        normalized_eigvects[y, x, 1],
                        ax,
                    )
        # display eigenvectors of the keypoint
        add_vector_to_ax(
            colormap,
            norm,
            eigvals[zoom_radius, zoom_radius, 0],
            zoom_radius,
            zoom_radius,
            normalized_eigvects[zoom_radius, zoom_radius, 0],
            ax,
        )

        # add red pixel on the keypoint, with variable size
        kp_factor = zoom_radius * 0.01
        ax.scatter(
            [zoom_radius],
            [zoom_radius],
            c="r",
            s=zoom_radius * kp_factor,
        )
        ax.set_title("eigenvectors")
        ax.axis("off")

        # add the blue to red colormap of the arrows
        # create a ScalarMappable with the same colormap and normalization as the arrows
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])

        return sm


def compare_directions(
    g_img1, g_img2, kp1, kp2, zoom_radius=15, figsize=(20, 10), dpi=600
):
    """
    g_img1, g_img2: grayscale images
    kp1, kp2: SIFT keypoints
    zoom_radius: radius of the zoomed area in pixels
    figsize: size of the figure
    dpi: resolution of the figure

    Compute the principal directions of the 2 images and display them side by side

    Return: the matplotlib figure
    """

    # create figure and ax
    fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # compute eigenvectors and add them to the ax
    sm1 = visualize_curvature_directions_ax_sm(g_img1, kp1, zoom_radius, ax=ax[0])
    sm2 = visualize_curvature_directions_ax_sm(g_img2, kp2, zoom_radius, ax=ax[1])

    # add the colorbar of the colormap of the arrows
    fig.colorbar(sm1, ax=ax[0], fraction=0.046, pad=0.04)
    fig.colorbar(sm2, ax=ax[1], fraction=0.046, pad=0.04)

    # add title to each subplot
    ax[0].set_title("Image 1")
    ax[1].set_title("Image 2")

    # add legend
    fig.suptitle(
        f"Principal direction near matched SIFT Keypoints \n zoom_radius = {zoom_radius}",
        fontsize=10,
    )

    return fig


def visualize_gradients(g_img, keypoint, zoom_radius, step_percentage=5):
    """
    Compute gradients of all pixels in a zoomed area around a keypoint.
    display with color shifting from white to red with increase of magnitude
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
        # Compute gradients for all pixels in subimage, (excluding a pixel border).
        border_size = 1
        h, w = sub_img.shape
        gradients = np.zeros((h, w, 2), dtype=np.float32)

        # Downsample the computations by taking 1 pixel every step in each direction, instead of all pixels
        step = (2 * zoom_radius + 1) * step_percentage / 100
        step = np.round(step).astype(int)

        for y in range(border_size, h - border_size, step):
            for x in range(border_size, w - border_size, step):
                # assert y,x do not belong to keypoint, because it is calculated afterwards
                if y != zoom_radius or x != zoom_radius:
                    gradients[y, x, :] = compute_gradient(sub_img, (y, x))
        # also compute gradient for the keypoint
        gradients[zoom_radius, zoom_radius, :] = compute_gradient(
            sub_img, (zoom_radius, zoom_radius)
        )

        # create a colormap for gradients, shifting from white to red with increase of magnitude
        colormap = plt.cm.get_cmap("Reds")

        # compute norm of gradients
        norms_gradients = np.linalg.norm(gradients, axis=-1)

        # normalize gradients with the min and max eigenvalues
        vmin, vmax = np.min(norms_gradients), np.max(norms_gradients)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        # normalize gradient so they have same length grad_size
        grad_size = zoom_radius * 0.05
        # and avoiding null gradients with a mask
        eps = 1e-6
        non_null_grad_mask = norms_gradients > eps

        unit_gradients = np.zeros_like(gradients, dtype=np.float32)

        # Create a new array for the reciprocal of the non-null norms
        reciprocal_norms = np.ones_like(norms_gradients)
        reciprocal_norms[non_null_grad_mask] = 1 / norms_gradients[non_null_grad_mask]

        # Normalize the non-null gradients
        unit_gradients[non_null_grad_mask] = (
            gradients[non_null_grad_mask]
            * reciprocal_norms[non_null_grad_mask, ..., np.newaxis]
            * grad_size
        )

        unit_gradients[~non_null_grad_mask] = 0.0

        # plot the subimage and the subimage with the gradients with colormap and variable size on it
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # plot subimage
        axs[0].imshow(sub_img, cmap="gray")
        axs[0].set_title(f"zoomed image, radius={zoom_radius}")
        axs[0].axis("off")
        # add red pixel on the keypoint
        kp_factor = 0.1
        axs[0].scatter([zoom_radius], [zoom_radius], c="r", s=zoom_radius * kp_factor)

        # plot the gradients on the subimage
        axs[1].imshow(sub_img, cmap="gray")
        for y in range(border_size, h - border_size, step):
            for x in range(border_size, w - border_size, step):
                # assert y,x do not belong to keypoint, because it is calculated afterwards
                if y != zoom_radius or x != zoom_radius:
                    add_vector_to_ax(
                        colormap,
                        norm,
                        norms_gradients[y, x],
                        x,
                        y,
                        unit_gradients[y, x],
                        axs[1],
                    )

        # display gradient of the keypoint
        add_vector_to_ax(
            colormap,
            norm,
            norms_gradients[zoom_radius, zoom_radius],
            zoom_radius,
            zoom_radius,
            unit_gradients[zoom_radius, zoom_radius],
            axs[1],
        )
        # add red pixel on the keypoint
        axs[1].scatter([zoom_radius], [zoom_radius], c="r", s=zoom_radius * kp_factor)
        axs[1].set_title(f"gradients, the reder, the bigger, radius={zoom_radius}")
        axs[1].axis("off")

        # create a ScalarMappable with the same colormap and normalization as the arrows
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])

        # add the colorbar of the colormap of the arrows
        fig.colorbar(sm, ax=axs[1], fraction=0.046, pad=0.04)

        # add legend
        fig.suptitle(f"SIFT Keypoint {y_kp}, {x_kp} (in red)", fontsize=10)

        return fig


def visualize_gradients_ax_sm(g_img, keypoint, zoom_radius, ax, step_percentage=5):
    """
    g_img: grayscale image
    keypoint: SIFT keypoint
    zoom_radius: radius of the zoomed area in pixels

    Compute gradients of all pixels in a zoomed area around a keypoint.
    display with color shifting from white to red with increase of magnitude
    Does nothing if the zoomed area is not in the image.
    Inplace modifies the matplotlib ax passed as argument

    Return: the matplotlib scalable colormap of the arrows
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
        # Compute gradients for all pixels in subimage, (excluding a pixel border).
        border_size = 1
        h, w = sub_img.shape
        gradients = np.zeros((h, w, 2), dtype=np.float32)

        # Downsample the computations by taking 1 pixel every step in each direction, instead of all pixels
        step = (2 * zoom_radius + 1) * step_percentage / 100
        step = np.round(step).astype(int)

        for y in range(border_size, h - border_size, step):
            for x in range(border_size, w - border_size, step):
                # assert y,x do not belong to keypoint, because it is calculated afterwards
                if y != zoom_radius or x != zoom_radius:
                    gradients[y, x, :] = compute_gradient(sub_img, (y, x))
        # also compute gradient for the keypoint
        gradients[zoom_radius, zoom_radius, :] = compute_gradient(
            sub_img, (zoom_radius, zoom_radius)
        )

        # create a colormap for gradients, shifting from white to red with increase of magnitude
        colormap = plt.cm.get_cmap("Reds")

        # compute norm of gradients
        norms_gradients = np.linalg.norm(gradients, axis=-1)

        # normalize gradients with the min and max eigenvalues
        vmin, vmax = np.min(norms_gradients), np.max(norms_gradients)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        # normalize gradient so they have same length grad_size
        grad_size = zoom_radius * 0.05
        # and avoiding null gradients with a mask
        eps = 1e-6
        non_null_grad_mask = norms_gradients > eps

        unit_gradients = np.zeros_like(gradients, dtype=np.float32)

        # Create a new array for the reciprocal of the non-null norms
        reciprocal_norms = np.ones_like(norms_gradients)
        reciprocal_norms[non_null_grad_mask] = 1 / norms_gradients[non_null_grad_mask]

        # Normalize the non-null gradients
        unit_gradients[non_null_grad_mask] = (
            gradients[non_null_grad_mask]
            * reciprocal_norms[non_null_grad_mask, ..., np.newaxis]
            * grad_size
        )

        unit_gradients[~non_null_grad_mask] = 0.0

        kp_factor = zoom_radius * 0.01

        # plot the gradients on the subimage
        ax.imshow(sub_img, cmap="gray")
        for y in range(border_size, h - border_size, step):
            for x in range(border_size, w - border_size, step):
                # assert y,x do not belong to keypoint, because it is calculated afterwards
                if y != zoom_radius or x != zoom_radius:
                    add_vector_to_ax(
                        colormap,
                        norm,
                        norms_gradients[y, x],
                        x,
                        y,
                        unit_gradients[y, x],
                        ax,
                    )

        # display gradient of the keypoint
        add_vector_to_ax(
            colormap,
            norm,
            norms_gradients[zoom_radius, zoom_radius],
            zoom_radius,
            zoom_radius,
            unit_gradients[zoom_radius, zoom_radius],
            ax,
        )
        # add red pixel on the keypoint
        ax.scatter([zoom_radius], [zoom_radius], c="r", s=zoom_radius * kp_factor)
        ax.set_title(f"gradients, the reder, the bigger, radius={zoom_radius}")
        ax.axis("off")

        # create a ScalarMappable with the same colormap and normalization as the arrows
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])

        return sm


def compare_gradients(
    g_img1, g_img2, kp1, kp2, zoom_radius=15, figsize=(20, 10), dpi=600
):
    """
    g_img1, g_img2: grayscale images
    kp1, kp2: SIFT keypoints
    zoom_radius: radius of the zoomed area in pixels
    figsize: size of the figure
    dpi: resolution of the figure

    Compute the gradients of the 2 images and display them side by side

    Return: the matplotlib figure
    """

    # create figure and ax
    fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # compute eigenvectors and add them to the ax
    sm1 = visualize_gradients_ax_sm(g_img1, kp1, zoom_radius, ax=ax[0])
    sm2 = visualize_gradients_ax_sm(g_img2, kp2, zoom_radius, ax=ax[1])

    # add the colorbar of the colormap of the arrows
    fig.colorbar(sm1, ax=ax[0], fraction=0.046, pad=0.04)
    fig.colorbar(sm2, ax=ax[1], fraction=0.046, pad=0.04)

    # add title to each subplot
    ax[0].set_title("Image 1")
    ax[1].set_title("Image 2")

    # add legend
    fig.suptitle(
        f"Gradients near matched SIFT Keypoints \n zoom_radius = {zoom_radius}",
        fontsize=10,
    )

    return fig


# TESTS
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    print("oui")

    # load grayscale image
    # img_path = "../images"
    im_name = "dumbbell"
    img_ext = "jpg"
    # img = cv.imread(f"{img_path}/{im_name}.{img_ext}", 0)
    img = cv.imread(f"{im_name}.{img_ext}", 0)
    # # show image
    # # plt.imshow(img, cmap="gray")
    # # plt.show()

    # define folder to save images
    img_folder = "zoomed_kp"
    img_resolution = 400  # in dpi

    # calculate sift keypoints and descriptors
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    # draw keypoints on image
    img_kp = cv.drawKeypoints(
        img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    nb_kp = len(keypoints)
    # plt.imshow(img_kp)
    # plt.title(f"{nb_kp} SIFT keypoints")
    # # # save figure
    # # plt.savefig(
    # #     f"{im_name}_sift.png",
    # #     dpi=img_resolution,
    # # )
    # plt.show()

    # draw colormap of eigenvalues of Hessian matrix for 1 keypoint
    kp0 = keypoints[200]
    y_kp0, x_kp0 = np.round(kp0.pt).astype(int)
    # print(kp0.pt)

    ###################################
    # Test visualize_curvature_values #

    eigval_fig = visualize_curvature_values(img, kp0, 30)

    # plt.figure(eigval_fig.number)
    # plt.show()

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

    # plt.figure(eig_fig.number)
    # plt.show()

    # # save figure
    # eig_fig.savefig(
    #     f"{img_folder}/zoomed_{im_name}_kp_{y_kp0}_{x_kp0}_{zoom_radius}_eigvects.png",
    #     dpi=img_resolution,
    # )

    # # test a bunch of keypoints
    # zoom_radius = 30
    # start_kp = 50
    # nb_kp = 5
    # step_idx = 50
    # end_kp = start_kp + nb_kp * step_idx
    # for kp in keypoints[start_kp:end_kp:step_idx]:
    #     eigvec_fig = visualize_curvature_directions(img, kp, zoom_radius)
    #     plt.figure(eigvec_fig.number)
    #     plt.show()

    ################
    # Test gradients
    ################

    grad_fig = visualize_gradients(img, kp0, zoom_radius)

    plt.figure(grad_fig.number)
    plt.show()

    ##############
    # test visualise_curvature_directions_ax_sm
    ##############

    # create figure and ax
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # compute eigenvectors and add them to the ax
    sm = visualize_curvature_directions_ax_sm(img, kp0, zoom_radius, ax=ax)

    # add the colorbar of the colormap of the arrows
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    # add legend
    fig.suptitle(
        f"SIFT Keypoint {y_kp0}, {x_kp0} (in red) from 2nd function", fontsize=10
    )

    plt.show()

    ##################
    # Test compare directions
    ##################

    # load grayscale image
    img_path = "../data"
    im_name1 = "cube-0"
    im_name2 = "cube-1"
    img_ext = "png"
    g_img1 = cv.imread(f"{img_path}/{im_name1}.{img_ext}", 0)
    g_img2 = cv.imread(f"{img_path}/{im_name2}.{img_ext}", 0)

    # calculate sift keypoints and descriptors
    sift = cv.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(g_img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(g_img2, None)

    # match descriptors
    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Get keypoints of first match
    kp1 = keypoints1[matches[0][0].queryIdx]
    kp2 = keypoints2[matches[0][0].trainIdx]

    # compare directions
    dir_fig = compare_directions(g_img1, g_img2, kp1, kp2, zoom_radius=15)

    # show figure
    plt.figure(dir_fig.number)
    plt.show()

    ##################
    # Test compare gradients
    ##################

    grad_fig = compare_gradients(g_img1, g_img2, kp1, kp2, zoom_radius=15)

    # show figure
    plt.figure(grad_fig.number)
    plt.show()
