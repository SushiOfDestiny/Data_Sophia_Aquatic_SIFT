import os
import sys

# sys.path.append("../PythonSIFT")
# import pysift

# # return to current directory
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# print(os.getcwd())

import cv2 as cv

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


def convert_uint8_to_float32(img):
    """
    Convert a uint8 image to float32 image.
    img: uint8 image
    Return: float32 image
    """
    # force cast into float32 to avoid overflow
    img = img.astype(np.float32)
    # normalize image between 0 and 1
    img = img / 255
    return img


def compute_gradient(g_img, position):
    """
    Compute gradient with finite differences at order 1.
    g_img: float32 grayscale image
    position: (y,x) position of point in pixels
    """
    y, x = position
    dx = g_img[y, x + 1] - g_img[y, x - 1]
    dy = g_img[y + 1, x] - g_img[y - 1, x]
    return np.array([dx, dy])


def compute_hessian(g_img, position):
    """
    Compute hessian with finite differences at order 2.
    g_img: float32 grayscale image
    position: (y,x) position of the pixel
    """
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


def crop_image_around_keypoint(g_img, position, zoom_radius):
    """
    Crop image around keypoint.
    g_img: float32 grayscale image
    position: (y,x) position of the keypoint
    zoom_radius: radius of the zoomed area in pixels
    Return None if the zoomed area is not in the image.
    Return the cropped image
    """
    y_kp, x_kp = position

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

        return sub_img

    else:
        print(
            f"Zoom around keypoint {y_kp}, {x_kp} of radius {zoom_radius} is not in the image"
        )


def compute_hessian_gradient_subimage(sub_img, border_size=1):
    """
    Compute hessian and gradient of neighbors of the keypoint within a square neighborhood.
    sub_img: float32 grayscale image
    border_size: size of the border to exclude from the computation
    Return None if the zoomed area is not in the image.
    Return array of hessian eigenvalues, array of hessian eigenvectors, array of gradients
    """

    h, w = sub_img.shape
    # initialize arrays
    eigvals = np.zeros((h, w, 2), dtype=np.float32)
    eigvects = np.zeros((h, w, 2, 2), dtype=np.float32)
    gradients = np.zeros((h, w, 2), dtype=np.float32)

    # loop over neighbors
    for y in range(border_size, h - border_size):
        for x in range(border_size, w - border_size):
            # simultaneously compute eigenvalues and eigenvectors
            H = compute_hessian(sub_img, (y, x))
            eigvals[y, x], eigvects[y, x] = np.linalg.eig(H)
            gradients[y, x] = compute_gradient(sub_img, (y, x))

    return eigvals, eigvects, gradients


# def visualize_curvature_values(g_img, keypoint, zoom_radius):
#     """
#     Compute eigenvalues of the Hessian matrix of all pixels in a zoomed area around a keypoint.
#     Does nothing if the zoomed area is not in the image.
#     g_img: grayscale image
#     keypoint: SIFT keypoint
#     zoom_radius: radius of the zoomed area in pixels
#     Return: the matplotlib figure
#     """
#     # compute pixel coordinates of the keypoint
#     y, x = keypoint.pt
#     y_kp = np.round(y).astype(int)
#     x_kp = np.round(x).astype(int)

#     # check if zoomed area is in the image
#     is_in_image = (
#         y_kp - zoom_radius >= 0
#         and y_kp + zoom_radius < g_img.shape[0]
#         and x_kp - zoom_radius >= 0
#         and x_kp + zoom_radius < g_img.shape[1]
#     )

#     if is_in_image:
#         # crop image around keypoint
#         sub_img = g_img[
#             y_kp - zoom_radius : y_kp + zoom_radius,
#             x_kp - zoom_radius : x_kp + zoom_radius,
#         ]
#         # Compute hessian eigenvalues of all pixels in subimage, (excluding a pixel border).
#         border_size = 1
#         h, w = sub_img.shape
#         eigvals = np.zeros((h, w, 2), dtype=np.float32)
#         for y in range(border_size, h - border_size):
#             for x in range(border_size, w - border_size):
#                 H = compute_hessian(sub_img, (y, x))
#                 eigvals[y, x] = np.linalg.eigvals(H)

#         # # Normalize eigenvalues with the max absolute value
#         max_abs_eigvals = np.max(np.abs(eigvals))
#         normalized_eigvals = eigvals / max_abs_eigvals
#         # normalized_eigvals = eigvals

#         # Compute colormap images
#         eigvals1 = normalized_eigvals[:, :, 0]
#         eigvals2 = normalized_eigvals[:, :, 1]

#         # Affine transform eigenvalues from [-1,1] to [0, 1]
#         eigvals1 = (eigvals1 + 1) / 2
#         eigvals2 = (eigvals2 + 1) / 2

#         # Plot subimage and eigenvalues
#         fig, axs = plt.subplots(1, 3, figsize=(15, 5))

#         # Plot subimage
#         im0 = axs[0].imshow(sub_img, cmap="gray")
#         axs[0].set_title(f"zoomed image, radius={zoom_radius}")
#         axs[0].axis("off")
#         # add red pixel on the keypoint
#         axs[0].scatter([zoom_radius], [zoom_radius], c="r")

#         # Plot colormap images
#         im1 = axs[1].imshow(eigvals1, cmap="gray", vmin=0, vmax=1)
#         # im1 = axs[1].imshow(eigvals1, cmap="gray", vmin=-1, vmax=1)
#         axs[1].set_title("eigenvalue 1")
#         axs[1].axis("off")
#         # add red pixel on the keypoint
#         axs[1].scatter([zoom_radius], [zoom_radius], c="r")

#         im2 = axs[2].imshow(eigvals2, cmap="gray", vmin=0, vmax=1)
#         # im2 = axs[2].imshow(eigvals2, cmap="gray", vmin=-1, vmax=1)
#         axs[2].set_title("eigenvalue 2")
#         axs[2].axis("off")
#         # add red pixel on the keypoint
#         axs[2].scatter([zoom_radius], [zoom_radius], c="r")

#         # # Define the same normalization for colorbars 1 and 2
#         # norm = colors.Normalize(vmin=0, vmax=1)

#         # Add colorbar and adjust its size so it matches the images
#         fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
#         fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
#         fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

#         # add legend
#         fig.suptitle(f"SIFT Keypoint {y_kp}, {x_kp} (in red)", fontsize=10)

#         # plt.tight_layout()
#         # plt.show()

#         return fig


def visualize_curvature_values(g_img, keypoint, zoom_radius):
    """
    Compute eigenvalues of the Hessian matrix of all pixels in a zoomed area around a keypoint.
    Does nothing if the zoomed area is not in the image.
    g_img: float32 grayscale image
    keypoint: SIFT keypoint
    zoom_radius: radius of the zoomed area in pixels
    Return: the matplotlib figure
    """
    # compute pixel coordinates of the keypoint
    y_kp, x_kp = np.round(keypoint.pt).astype(int)

    # try to crop subimage around keypoint
    sub_img = crop_image_around_keypoint(g_img, (y_kp, x_kp), zoom_radius)

    if sub_img is not None:
        # Compute hessian eigenvalues
        eigvals, _, _ = compute_hessian_gradient_subimage(sub_img)

        # Normalize eigenvalues with the max absolute value
        max_abs_eigvals = np.max(np.abs(eigvals))
        normalized_eigvals = eigvals / max_abs_eigvals

        # Compute colormap images
        eigvals1 = normalized_eigvals[:, :, 0]
        eigvals2 = normalized_eigvals[:, :, 1]

        # Affine transform eigenvalues from [-1,1] to [0, 1]
        eigvals1 = (eigvals1 + 1) / 2
        eigvals2 = (eigvals2 + 1) / 2

        # Plot subimage and eigenvalues
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Define the images and titles
        images = [sub_img, eigvals1, eigvals2]
        titles = [f"zoomed image, radius={zoom_radius}", "eigenvalue 1", "eigenvalue 2"]
        v_min_max = [(None, None), (0, 1), (0, 1)]

        # Loop over the axes, images, and titles
        for ax, image, title in zip(axs, images, titles):
            vmin, vmax = v_min_max.pop(0)
            # Plot the image
            im = ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")

            # Add a red pixel on the keypoint
            ax.scatter([zoom_radius], [zoom_radius], c="r")

            # Add a colorbar and adjust its size so it matches the image
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # add legend
        fig.suptitle(f"SIFT Keypoint {y_kp}, {x_kp} (in red)", fontsize=10)

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


def compute_downsampling_step(step_percentage, zoom_radius):
    """
    Compute the downsampling step from a percentage of pixels to skip.
    step_percentage: percentage of pixels to skip in the computation of the eigenvectors, w.r.t subimage size
    zoom_radius: radius of the zoomed area in pixels
    Return: the downsampling step
    """
    step = (2 * zoom_radius + 1) * step_percentage / 100
    step = np.round(step).astype(int)
    return step


def visualize_curvature_directions(g_img, keypoint, zoom_radius, step_percentage=5):
    """
    Compute eigenvectors of the Hessian matrix of all pixels in a zoomed area around a keypoint.
    display the directions in 2 colors depending on the sign of the eigenvalues.
    Does nothing if the zoomed area is not in the image.
    g_img: float32 grayscale image
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
        step = compute_downsampling_step(step_percentage, zoom_radius)

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


# def visualize_curvature_directions_ax_sm(
#     g_img, keypoint, zoom_radius, ax, step_percentage=5
# ):
#     """
#     g_img: float32 grayscale image
#     keypoint: SIFT keypoint
#     zoom_radius: radius of the zoomed area in pixels
#     step_percentage: percentage of pixels to skip in the computation of the eigenvectors, w.r.t subimage size
#     ax: matplotlib axis

#     Compute eigenvectors of the Hessian matrix of all pixels in a zoomed area around a keypoint.
#     display the directions in 2 colors depending on the sign of the eigenvalues.
#     Does nothing if the zoomed area is not in the image.
#     Inplace modify the argument ax

#     Return: the scalable colormap of the arrows
#     """
#     # compute pixel coordinates of the keypoint
#     y, x = keypoint.pt
#     y_kp = np.round(y).astype(int)
#     x_kp = np.round(x).astype(int)

#     # check if zoomed area is in the image
#     is_in_image = (
#         y_kp - zoom_radius >= 0
#         and y_kp + zoom_radius < g_img.shape[0]
#         and x_kp - zoom_radius >= 0
#         and x_kp + zoom_radius < g_img.shape[1]
#     )

#     if not is_in_image:
#         print("zoomed area is not fully in the image")
#     else:
#         # crop image around keypoint
#         sub_img = g_img[
#             y_kp - zoom_radius : y_kp + zoom_radius,
#             x_kp - zoom_radius : x_kp + zoom_radius,
#         ]
#         # Compute hessian eigenvectors and eigenvalues of all pixels in subimage, (excluding a pixel border).
#         border_size = 1
#         h, w = sub_img.shape
#         eigvects = np.zeros((h, w, 2, 2), dtype=np.float32)
#         eigvals = np.zeros((h, w, 2), dtype=np.float32)

#         # Downsample the computations by taking 1 pixel every step in each direction, instead of all pixels
#         step = (2 * zoom_radius + 1) * step_percentage / 100
#         step = np.round(step).astype(int)

#         for y in range(border_size, h - border_size, step):
#             for x in range(border_size, w - border_size, step):
#                 # assert y,x do not belong to keypoint, because it is calculated afterwards
#                 if y != zoom_radius or x != zoom_radius:
#                     H = compute_hessian(sub_img, (y, x))
#                     # simultaneously compute eigenvalues and eigenvectors
#                     eigvals[y, x], eigvects[y, x] = np.linalg.eig(H)
#         # also compute eigenvalues and eigenvectors for the keypoint
#         H_kp = compute_hessian(sub_img, (zoom_radius, zoom_radius))
#         (
#             eigvals[zoom_radius, zoom_radius],
#             eigvects[zoom_radius, zoom_radius],
#         ) = np.linalg.eig(H_kp)

#         # # normalize eigenvectors with the max absolute value
#         # norms_eigvects = np.linalg.norm(eigvects, axis=-1)
#         # max_abs_eigvects = np.max(norms_eigvects)
#         # print("max_abs_eigvects 1", max_abs_eigvects)
#         # normalized_eigvects = eigvects / max_abs_eigvects
#         normalized_eigvects = eigvects

#         # plot eigenvectors
#         # define colormap for eigenvectors depending on the value of the eigenvalues
#         # the higher the eigenvalue, the more red the eigenvector, the lower the eigenvalue, the more blue the eigenvector
#         colormap = plt.cm.get_cmap("RdBu")

#         # normalize eigenvalues with the min and max eigenvalues
#         vmin, vmax = np.min(eigvals), np.max(eigvals)
#         norm = colors.Normalize(vmin=vmin, vmax=vmax)

#         # plot eigenvectors on another image
#         ax.imshow(sub_img, cmap="gray")

#         for y in range(border_size, h - border_size, step):
#             for x in range(border_size, w - border_size, step):
#                 # assert y,x do not belong to keypoint, because it is calculated afterwards
#                 if y != zoom_radius or x != zoom_radius:
#                     add_vector_to_ax(
#                         colormap,
#                         norm,
#                         eigvals[y, x, 0],
#                         x,
#                         y,
#                         normalized_eigvects[y, x, 0],
#                         ax,
#                     )
#                     add_vector_to_ax(
#                         colormap,
#                         norm,
#                         eigvals[y, x, 1],
#                         x,
#                         y,
#                         normalized_eigvects[y, x, 1],
#                         ax,
#                     )
#         # display eigenvectors of the keypoint
#         add_vector_to_ax(
#             colormap,
#             norm,
#             eigvals[zoom_radius, zoom_radius, 0],
#             zoom_radius,
#             zoom_radius,
#             normalized_eigvects[zoom_radius, zoom_radius, 0],
#             ax,
#         )
#         add_vector_to_ax(
#             colormap,
#             norm,
#             eigvals[zoom_radius, zoom_radius, 1],
#             zoom_radius,
#             zoom_radius,
#             normalized_eigvects[zoom_radius, zoom_radius, 1],
#             ax,
#         )

#         # add red pixel on the keypoint, with variable size
#         kp_factor = zoom_radius * 0.01
#         ax.scatter(
#             [zoom_radius],
#             [zoom_radius],
#             c="r",
#             s=zoom_radius * kp_factor,
#         )
#         ax.set_title("eigenvectors")
#         ax.axis("off")

#         # add the blue to red colormap of the arrows
#         # create a ScalarMappable with the same colormap and normalization as the arrows
#         sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
#         sm.set_array([])

#         return sm


def downsample_array(array, step, border_size=1):
    """
    Downsample an array by taking 1 pixel every step in each direction, instead of all pixels,
    except for a border.
    array: array to downsample
    step: step of the downsampling
    border_size: size of the border to exclude from the computation
    Return: the downsampled array
    """
    # Create a mask for the first 2 dimensions
    # mask = [slice(0, -1, step), slice(0, -1, step)]
    mask = (
        slice(border_size, -border_size, step),
        slice(border_size, -border_size, step),
    )

    selected_array = np.zeros_like(array, dtype=np.float32)
    selected_array[mask] = array[mask]

    return selected_array


def draw_vectors_on_ax(
    ax,
    colormap,
    norm,
    cmap_values,
    vects,
    im_shape,
    step,
    border_size=1,
):
    """
    Add vectors to an axis with a colormap depending on a value.
    ax: matplotlib axis
    colormap: matplotlib colormap
    norm: matplotlib norm
    cmap_values: values to get the color from the colormap
    vects: vectors to plot
    im_shape: shape of the image, (h, w)
    step: step of the downsampling
    border_size: size of the border to exclude from the computation
    """
    h, w = im_shape

    for y in range(border_size, h - border_size, step):
        for x in range(border_size, w - border_size, step):
            add_vector_to_ax(
                colormap,
                norm,
                cmap_values[y, x],
                x,
                y,
                vects[y, x],
                ax,
            )


def normalize_vectors_2D_array(array, eps=1e-8):
    """
    Normalize non null vectors of a 2D array.
    array: 2D array of N-dimensional vectors, shape (h, w, N)
    eps: threshold for null values
    """
    # compute norm of vectors
    norms = np.linalg.norm(array, axis=-1)
    # avoid division by 0
    norms[norms <= eps] = 1
    # normalize vectors
    normalized_array = array / norms[..., np.newaxis]
    return normalized_array


def visualize_curvature_directions_ax_sm_unfinished(
    g_img, keypoint, zoom_radius, ax, step_percentage=5, border_size=1
):
    """
    g_img: float32 grayscale image
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

    # try to crop image around keypoint
    sub_img = crop_image_around_keypoint(g_img, (y_kp, x_kp), zoom_radius)
    h, w = sub_img.shape

    # Compute hessian eigenvectors and eigenvalues of all pixels in subimage.
    eigvals, eigvects, _ = compute_hessian_gradient_subimage(sub_img)

    # Downsample the computations by taking 1 pixel every step in each direction, instead of all pixels
    # We downsample within pixels that are not in the border
    # Values of unconsidered pixels are set to 0
    step = compute_downsampling_step(step_percentage, zoom_radius)
    selected_eigvects = downsample_array(eigvects, step)
    selected_eigvals = downsample_array(eigvals, step)
    # Add values of the keypoint
    selected_eigvects[zoom_radius, zoom_radius] = eigvects[zoom_radius, zoom_radius]
    selected_eigvals[zoom_radius, zoom_radius] = eigvals[zoom_radius, zoom_radius]

    # normalize eigenvectors
    vect_size = zoom_radius * 0.05
    normalized_eigvects = normalize_vectors_2D_array(selected_eigvects)
    normalized_eigvects *= vect_size

    # draw eigenvectors on ax
    # define colormap for eigenvectors depending on the value of the eigenvalues
    # the higher the eigenvalue, the more red the eigenvector, the lower the eigenvalue, the more blue the eigenvector
    colormap = plt.cm.get_cmap("RdBu")

    # normalize eigenvalues with the min and max eigenvalues
    vmin, vmax = np.min(selected_eigvals), np.max(selected_eigvals)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # draw eigenvectors on ax
    ax.imshow(sub_img, cmap="gray")

    # draw eigenvectors of all pixels
    for i in range(2):
        # first avoid keypoint
        draw_vectors_on_ax(
            ax,
            colormap,
            norm,
            selected_eigvals[:, :, i],
            normalized_eigvects[:, :, i],
            (h, w),
            step,
            border_size,
        )

    # add red pixel on the keypoint, with variable size
    kp_factor = zoom_radius * 0.1
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


def visualize_gradients_ax_sm_unfinished(
    g_img, keypoint, zoom_radius, ax, step_percentage=5, border_size=1
):
    """
    g_img: grayscale image
    keypoint: SIFT keypoint
    zoom_radius: radius of the zoomed area in pixels
    ax: matplotlib axis
    step_percentage: percentage of pixels to skip in the computation of the eigenvectors, w.r.t subimage size
    border_size: size of the border to exclude from the computation

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

    # try to crop image around keypoint
    sub_img = crop_image_around_keypoint(g_img, (y_kp, x_kp), zoom_radius)
    h, w = sub_img.shape

    # Compute gradients for all pixels in subimage
    _, _, gradients = compute_hessian_gradient_subimage(sub_img)

    # Downsample the computations by taking 1 pixel every step in each direction, instead of all pixels
    # Values of unconsidered pixels are set to 0
    step = compute_downsampling_step(step_percentage, zoom_radius)
    selected_gradients = downsample_array(gradients, step, border_size)
    # Add values of the keypoint
    selected_gradients[zoom_radius, zoom_radius] = gradients[zoom_radius, zoom_radius]

    # create a colormap for gradients norms, shifting from white to red with increase of magnitude
    colormap = plt.cm.get_cmap("Reds")
    norms_gradients = np.linalg.norm(selected_gradients, axis=-1)
    vmin, vmax = np.min(norms_gradients), np.max(norms_gradients)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # normalize gradient so they have same length grad_size
    normalized_gradients = normalize_vectors_2D_array(selected_gradients)
    grad_size = zoom_radius * 0.05
    normalized_gradients *= grad_size

    # plot the gradients on the subimage
    ax.imshow(sub_img, cmap="gray")
    draw_vectors_on_ax(
        ax,
        colormap,
        norm,
        norms_gradients,
        normalized_gradients,
        (h, w),
        step,
        border_size,
    )

    # add red pixel on the keypoint
    kp_factor = zoom_radius * 0.01
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
