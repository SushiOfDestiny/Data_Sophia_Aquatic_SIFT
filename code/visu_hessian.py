import os
import sys
import cv2 as cv

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

# add the path to the descriptor folder
sys.path.append(os.path.join("..", "descriptor"))
# import visualize_hessian.visu_hessian
import descriptor as desc

# return to the root directory
sys.path.append(os.path.join(".."))


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
    position: (x,y) position of point in pixels
    return: gradient vector [dx, dy]
    """
    x, y = position
    dx = g_img[y, x + 1] - g_img[y, x - 1]
    dy = g_img[y + 1, x] - g_img[y - 1, x]
    return np.array([dx, dy])


def compute_hessian(g_img, position):
    """
    Compute hessian with finite differences at order 2.
    g_img: float32 grayscale image
    position: (x,y) position of the pixel
    return: hessian matrix [[dxx, dxy], [dxy, dyy]]
    """
    x, y = position
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
    position: (x,y) position of the keypoint
    zoom_radius: radius of the zoomed area in pixels
    Pixels at x_kp + radius are included in the zoomed area.
    Return None if the zoomed area is not in the image.
    Return the cropped image
    """
    x_kp, y_kp = position

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
            y_kp - zoom_radius : y_kp + zoom_radius + 1,
            x_kp - zoom_radius : x_kp + zoom_radius + 1,
        ]

        return sub_img

    else:
        print(
            f"Zoom around keypoint x:{x_kp}, y:{y_kp} of radius {zoom_radius} is not in the image"
        )


def compute_hessian_gradient_subimage(sub_img, border_size=1):
    """
    Compute hessian and gradient of neighbors of the keypoint within a square neighborhood.
    sub_img: float32 grayscale image
    border_size: size of the border to exclude from the computation
    For locations in the avoided border, the values are set to 0
    Eigenvalues and eigenvectors are in decreasing order of signed value
    Convention: the eigenvectors are the rows of the eigvects array,
    ie eigvects[y, x, 0, 1] is the vertical coord of the first eigenvector
    That is not the numpy convention.
    Return array of hessian eigenvalues, array of hessian eigenvectors, array of gradients, computed within the border
    Return None if the zoomed area is not in the image.
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
            H = compute_hessian(sub_img, (x, y))
            eigvals[y, x], eigvects_col = np.linalg.eig(
                H
            )  # numpy naturally returns the eigenvectors as columns
            # sort eigenvalues and eigenvectors in decreasing order of signed value
            # numpy argsort by default sorts in increasing order, so we pass the opposite of the eigenvalues
            idx = np.argsort(-eigvals[y, x])
            eigvals[y, x] = eigvals[y, x, idx]
            eigvects_col = eigvects_col[:, idx]

            # eigvect_col is a 2*2 array, each column is an eigenvector
            eigvects[
                y, x
            ] = eigvects_col.T  # we want the eigenvectors as rows (convention)
            gradients[y, x] = compute_gradient(sub_img, (x, y))

    return eigvals, eigvects, gradients


def visualize_curvature_values(g_img, keypoint, zoom_radius, figsize=(30, 10)):
    """
    Compute eigenvalues of the Hessian matrix of all pixels in a zoomed area around a keypoint.
    Does nothing if the zoomed area is not in the image.
    g_img: float32 grayscale image
    keypoint: SIFT keypoint
    zoom_radius: radius of the zoomed area in pixels
    Return: the matplotlib figure
    """
    # compute pixel coordinates of the keypoint
    x_kp, y_kp = np.round(keypoint.pt).astype(int)

    # try to crop subimage around keypoint
    sub_img = crop_image_around_keypoint(g_img, (x_kp, y_kp), zoom_radius)

    if sub_img is not None:
        # Compute hessian eigenvalues
        eigvals, _, _ = compute_hessian_gradient_subimage(sub_img)

        # # Normalize eigenvalues with the max absolute value
        # max_abs_eigvals = np.max(np.abs(eigvals))
        # normalized_eigvals = eigvals / max_abs_eigvals

        # # Compute colormap images
        # eigvals1 = normalized_eigvals[:, :, 0]
        # eigvals2 = normalized_eigvals[:, :, 1]

        # # Affine transform eigenvalues from [-1,1] to [0, 1]
        # eigvals1 = (eigvals1 + 1) / 2
        # eigvals2 = (eigvals2 + 1) / 2

        # Plot subimage and eigenvalues
        fig, axs = plt.subplots(1, 3, figsize=figsize)

        # compute extrema of eigvals
        eigvals1 = eigvals[:, :, 0]
        eigvals2 = eigvals[:, :, 1]
        vmins = [np.min(eigvals1), np.min(eigvals2)]
        vmaxs = [np.max(eigvals1), np.max(eigvals2)]

        # Define the images and titles
        images = [sub_img, eigvals1, eigvals2]
        titles = [f"zoomed image on keypoint", "eigenvalue 1", "eigenvalue 2"]
        v_min_max = [(None, None), (vmins[0], vmaxs[0]), (vmins[1], vmaxs[1])]

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
        fig.suptitle(
            f"SIFT Keypoint x:{x_kp}, y:{y_kp} (in red) \n radius={zoom_radius}, unnormalized values",
            fontsize=10,
        )

        return fig


# def rotate_subimage(g_img, x_kp, y_kp, orientation, zoom_radius):
#     # define bigger radius for the subimage to crop to make sure all pixels to rotate are in the cropped image
#     bigger_radius = 2 * int(0.5 * np.ceil(zoom_radius * np.sqrt(2))) + 1

#     # try to crop subimage around keypoint
#     big_sub_img = crop_image_around_keypoint(g_img, (x_kp, y_kp), bigger_radius)

#     # define the coordinates of center of the bigger subimage
#     x_big_sub_img_center, y_big_sub_img_center = 2 * bigger_radius, 2 * bigger_radius

#     # put in a new array the rotated points of the square of radius zoom_radius
#     rotated_big_sub_image = np.zero_like(
#         big_sub_img, dtype=np.float32
#     )
#     # loop over cols and rows with the coordinates in the bigger subimage
#     for i in range(
#         y_big_sub_img_center - zoom_radius, y_big_sub_img_center + zoom_radius + 1
#     ):
#         for j in range(
#             x_big_sub_img_center - zoom_radius, x_big_sub_img_center + zoom_radius + 1
#         ):
#             # rotate pixel
#             rot_i, rot_j = desc.rotate_point_pixel(
#                 i,
#                 j,
#                 y_big_sub_img_center,
#                 x_big_sub_img_center,
#                 -orientation,
#             )
#             # put pixel in the new array
#             rotated_big_sub_image[rot_i, rot_j] = big_sub_img[i, j]

#     return rotated_big_sub_image


def rotate_subimage(g_img, x_kp, y_kp, orientation, zoom_radius):
    # define bigger radius for the subimage to crop to make sure all pixels to rotate are in the cropped image
    bigger_radius = 2 * int(0.5 * np.ceil(zoom_radius * np.sqrt(2))) + 1

    # try to crop subimage around keypoint
    big_sub_img = crop_image_around_keypoint(g_img, (x_kp, y_kp), bigger_radius)

    # define the coordinates of center of the bigger subimage
    x_big_sub_img_center, y_big_sub_img_center = bigger_radius, bigger_radius

    # create a new array for the rotated image
    rotated_big_sub_image = np.zeros_like(big_sub_img, dtype=np.float32)

    # loop over cols and rows with the coordinates in the bigger subimage
    y_start = y_big_sub_img_center - zoom_radius
    y_end = y_big_sub_img_center + zoom_radius + 1
    x_start = x_big_sub_img_center - zoom_radius
    x_end = x_big_sub_img_center + zoom_radius + 1
    for i in range(y_start, y_end):
        for j in range(x_start, x_end):
            # rotate pixel
            rot_i, rot_j = desc.rotate_point_pixel(
                i, j, -orientation, y_big_sub_img_center, x_big_sub_img_center
            )

            # check if the rotated coordinates are within the bounds of the new image array
            if (
                0 <= rot_i < rotated_big_sub_image.shape[0]
                and 0 <= rot_j < rotated_big_sub_image.shape[1]
            ):
                # put pixel in the new array
                rotated_big_sub_image[i, j] = big_sub_img[rot_i, rot_j]

    return rotated_big_sub_image[y_start:y_end, x_start:x_end]


def visualize_curvature_values_rotated(
    g_img, keypoint, orientation, zoom_radius, figsize=(30, 10)
):
    """
    Compute eigenvalues of the Hessian matrix of all pixels in a zoomed area around a keypoint.
    The area is rotated by the orientation of the keypoint in counterclockwise direction.
    Does nothing if the zoomed area is not in the image.
    g_img: float32 grayscale image
    keypoint: SIFT keypoint
    zoom_radius: radius of the zoomed area in pixels
    Return: the matplotlib figure
    """
    # compute pixel coordinates of the keypoint
    x_kp, y_kp = np.round(keypoint.pt).astype(int)

    # crop subimage around keypoint
    rot_sub_img = rotate_subimage(g_img, x_kp, y_kp, orientation, zoom_radius)

    if rot_sub_img is not None:
        # Compute hessian eigenvalues
        eigvals, _, _ = compute_hessian_gradient_subimage(rot_sub_img)

        # Plot subimage and eigenvalues
        fig, axs = plt.subplots(1, 3, figsize=figsize)

        # compute extrema of eigvals
        eigvals1 = eigvals[:, :, 0]
        eigvals2 = eigvals[:, :, 1]
        vmins = [np.min(eigvals1), np.min(eigvals2)]
        vmaxs = [np.max(eigvals1), np.max(eigvals2)]

        # Define the images and titles
        images = [rot_sub_img, eigvals1, eigvals2]
        titles = [f"zoomed image on keypoint", "eigenvalue 1", "eigenvalue 2"]
        v_min_max = [(None, None), (vmins[0], vmaxs[0]), (vmins[1], vmaxs[1])]

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
        fig.suptitle(
            f"Rotated image around SIFT Keypoint x:{x_kp}, y:{y_kp} (in red) \n radius={zoom_radius}, unnormalized values, rotation={orientation:.2f}°",
            fontsize=10,
        )

        return fig


def add_vector_to_ax(colormap, norm, cm_value, x, y, vect, ax, width=0.1):
    """
    Add a vector to an axis with a colormap depending on a value.
    colormap: matplotlib colormap
    norm: matplotlib norm
    cm_value: value to get the color from the colormap
    x, y: position of the vector
    vect: vector to plot, coordinates are vect[0]=x, vect[1]=y
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


def downsample_array(array, step, border_size=1):
    """
    Downsample an array by taking 1 pixel every step in each direction (the 2 first dimensions), instead of all pixels,
    except for a border.
    Non taken position have null values
    array: array to downsample, shape (h, w, ...)
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
    Divide all vectors of a 2D array by their norms.
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


def visualize_curvature_directions_ax_sm(
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
    x_kp, y_kp = keypoint.pt
    y_kp = np.round(y_kp).astype(int)
    x_kp = np.round(x_kp).astype(int)

    # try to crop image around keypoint
    sub_img = crop_image_around_keypoint(g_img, (x_kp, y_kp), zoom_radius)
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
    normalized_eigvects = np.zeros_like(selected_eigvects, dtype=np.float32)
    for eigvect_id in range(2):
        normalized_eigvects[:, :, eigvect_id] = normalize_vectors_2D_array(
            selected_eigvects[:, :, eigvect_id]
        )
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
    for eigvect_id in range(2):
        # first avoid keypoint
        draw_vectors_on_ax(
            ax,
            colormap,
            norm,
            selected_eigvals[:, :, eigvect_id],
            normalized_eigvects[:, :, eigvect_id],
            (h, w),
            step,
            border_size,
        )
        # then draw keypoint
        add_vector_to_ax(
            colormap,
            norm,
            selected_eigvals[zoom_radius, zoom_radius, eigvect_id],
            zoom_radius,
            zoom_radius,
            normalized_eigvects[zoom_radius, zoom_radius, eigvect_id],
            ax,
        )

    # add red pixel on the keypoint, with variable size
    kp_factor = zoom_radius * 0.05
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


def visualize_curvature_directions_ax_sm_rotated(
    g_img, keypoint, zoom_radius, ax, orientation, step_percentage=5, border_size=1
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
    x_kp, y_kp = keypoint.pt
    y_kp = np.round(y_kp).astype(int)
    x_kp = np.round(x_kp).astype(int)

    # try to crop image around keypoint
    sub_img = rotate_subimage(g_img, x_kp, y_kp, orientation, zoom_radius)
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
    normalized_eigvects = np.zeros_like(selected_eigvects, dtype=np.float32)
    for eigvect_id in range(2):
        normalized_eigvects[:, :, eigvect_id] = normalize_vectors_2D_array(
            selected_eigvects[:, :, eigvect_id]
        )
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
    for eigvect_id in range(2):
        # first avoid keypoint
        draw_vectors_on_ax(
            ax,
            colormap,
            norm,
            selected_eigvals[:, :, eigvect_id],
            normalized_eigvects[:, :, eigvect_id],
            (h, w),
            step,
            border_size,
        )
        # then draw keypoint
        add_vector_to_ax(
            colormap,
            norm,
            selected_eigvals[zoom_radius, zoom_radius, eigvect_id],
            zoom_radius,
            zoom_radius,
            normalized_eigvects[zoom_radius, zoom_radius, eigvect_id],
            ax,
        )

    # add red pixel on the keypoint, with variable size
    kp_factor = zoom_radius * 0.05
    ax.scatter(
        [zoom_radius],
        [zoom_radius],
        c="r",
        s=zoom_radius * kp_factor,
    )
    ax.set_title(f"eigenvectors rotated by {orientation:.2f}°")
    ax.axis("off")

    # add the blue to red colormap of the arrows
    # create a ScalarMappable with the same colormap and normalization as the arrows
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])

    return sm


def compare_directions(
    g_img1,
    g_img2,
    kp1,
    kp2,
    zoom_radius=15,
    figsize=(20, 10),
    dpi=600,
    step_percentage=5,
    border_size=1,
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
    sm1 = visualize_curvature_directions_ax_sm(
        g_img1, kp1, zoom_radius, ax[0], step_percentage, border_size
    )
    sm2 = visualize_curvature_directions_ax_sm(
        g_img2, kp2, zoom_radius, ax[1], step_percentage, border_size
    )

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


def compare_directions_rotated(
    g_img1,
    g_img2,
    kp1,
    kp2,
    orientations,
    zoom_radius=15,
    figsize=(20, 10),
    dpi=600,
    step_percentage=5,
    border_size=1,
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
    sm1 = visualize_curvature_directions_ax_sm_rotated(
        g_img1, kp1, zoom_radius, ax[0], orientations[0], step_percentage, border_size
    )
    sm2 = visualize_curvature_directions_ax_sm_rotated(
        g_img2, kp2, zoom_radius, ax[1], orientations[1], step_percentage, border_size
    )

    # add the colorbar of the colormap of the arrows
    fig.colorbar(sm1, ax=ax[0], fraction=0.046, pad=0.04)
    fig.colorbar(sm2, ax=ax[1], fraction=0.046, pad=0.04)

    # add title to each subplot
    ax[0].set_title("Image 1")
    ax[1].set_title("Image 2")

    # add legend
    fig.suptitle(
        f"Rotated principal direction near matched SIFT Keypoints \n zoom_radius = {zoom_radius}",
        fontsize=10,
    )

    return fig


def visualize_gradients_ax_sm(
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
    x_kp, y_kp = keypoint.pt
    y_kp = np.round(y_kp).astype(int)
    x_kp = np.round(x_kp).astype(int)

    # try to crop image around keypoint
    sub_img = crop_image_around_keypoint(g_img, (x_kp, y_kp), zoom_radius)
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
    # first avoid keypoint
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
    # then draw keypoint
    add_vector_to_ax(
        colormap,
        norm,
        norms_gradients[zoom_radius, zoom_radius],
        zoom_radius,
        zoom_radius,
        normalized_gradients[zoom_radius, zoom_radius],
        ax,
    )

    # add red pixel on the keypoint
    kp_factor = zoom_radius * 0.05
    ax.scatter([zoom_radius], [zoom_radius], c="r", s=zoom_radius * kp_factor)
    ax.set_title(f"gradients, the reder, the bigger, radius={zoom_radius}")
    ax.axis("off")

    # create a ScalarMappable with the same colormap and normalization as the arrows
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])

    return sm


def visualize_gradients_ax_sm_rotated(
    g_img, keypoint, zoom_radius, ax, orientation, step_percentage=5, border_size=1
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
    x_kp, y_kp = keypoint.pt
    y_kp = np.round(y_kp).astype(int)
    x_kp = np.round(x_kp).astype(int)

    # try to crop image around keypoint
    sub_img = rotate_subimage(g_img, x_kp, y_kp, orientation, zoom_radius)
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
    # first avoid keypoint
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
    # then draw keypoint
    add_vector_to_ax(
        colormap,
        norm,
        norms_gradients[zoom_radius, zoom_radius],
        zoom_radius,
        zoom_radius,
        normalized_gradients[zoom_radius, zoom_radius],
        ax,
    )

    # add red pixel on the keypoint
    kp_factor = zoom_radius * 0.05
    ax.scatter([zoom_radius], [zoom_radius], c="r", s=zoom_radius * kp_factor)
    ax.set_title(
        f"gradients, the reder, the bigger, radius={zoom_radius}, rotation={orientation:.2f}°"
    )
    ax.axis("off")

    # create a ScalarMappable with the same colormap and normalization as the arrows
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])

    return sm


def compare_gradients(
    g_img1,
    g_img2,
    kp1,
    kp2,
    zoom_radius=15,
    figsize=(20, 10),
    dpi=600,
    step_percentage=5,
    border_size=1,
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
    sm1 = visualize_gradients_ax_sm(
        g_img1, kp1, zoom_radius, ax[0], step_percentage, border_size
    )
    sm2 = visualize_gradients_ax_sm(
        g_img2, kp2, zoom_radius, ax[1], step_percentage, border_size
    )

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


def compare_gradients_rotated(
    g_img1,
    g_img2,
    kp1,
    kp2,
    orientations,
    zoom_radius=15,
    figsize=(20, 10),
    dpi=600,
    step_percentage=5,
    border_size=1,
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
    sm1 = visualize_gradients_ax_sm_rotated(
        g_img1, kp1, zoom_radius, ax[0], orientations[0], step_percentage, border_size
    )
    sm2 = visualize_gradients_ax_sm_rotated(
        g_img2, kp2, zoom_radius, ax[1], orientations[1], step_percentage, border_size
    )

    # add the colorbar of the colormap of the arrows
    fig.colorbar(sm1, ax=ax[0], fraction=0.046, pad=0.04)
    fig.colorbar(sm2, ax=ax[1], fraction=0.046, pad=0.04)

    # add title to each subplot
    ax[0].set_title("Image 1")
    ax[1].set_title("Image 2")

    # add legend
    fig.suptitle(
        f"Rotated Gradients near matched SIFT Keypoints \n zoom_radius = {zoom_radius}",
        fontsize=10,
    )

    return fig
