import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, fftconvolve
import scipy.ndimage as ndimage

# add the path to the visualize_hessian folder
sys.path.append(os.path.join("..", "visualize_hessian"))
# import visualize_hessian.visu_hessian
import visu_hessian as vh

# return to the root directory
sys.path.append(os.path.join(".."))

################
# Orientations #
################


def compute_orientation(g_img, position, epsilon=1e-6):
    """
    Compute the orientation of the keypoint with Lowe's formula.
    g_img: float32 grayscale image
    position: (x, y) int pixel position of point
    epsilon: float to avoid division by 0 (default is 1e-6)
    return orientation: float32 orientation of the keypoint in degrees, within [0, 360[
    """
    x, y = position
    denominator = g_img[y, x + 1] - g_img[y, x - 1]
    # if denominator is too small, add or subtract epsilon to avoid division by 0
    if np.abs(denominator) < epsilon:
        if denominator >= 0:
            denominator += epsilon
        else:
            denominator -= epsilon
    slope = (g_img[y + 1, x] - g_img[y - 1, x]) / denominator
    orientation = np.arctan(slope)

    return orientation


def convert_angles_to_pos_degrees(angles):
    """
    Convert array of angles in radians to positive degrees in [0, 360[
    """
    # translate the angles in [0, 2pi[
    posdeg_angles = angles % (2 * np.pi)
    # translate the orientation in [0, 360[
    posdeg_angles = posdeg_angles * 180 / np.pi
    return posdeg_angles


def compute_orientations(g_img, border_size=1):
    """
    Compute the orientation of all pixels of a grayscale image within a border
    g_img: float32 grayscale image
    border_size: int size of the border to ignore (default is 1)
    return orientations: float32 array of shape (height, width) of the orientations in radians
    """
    orientations = np.zeros_like(g_img, dtype=np.float32)
    for y in range(border_size, g_img.shape[0] - border_size):
        for x in range(border_size, g_img.shape[1] - border_size):
            orientations[y, x] = compute_orientation(g_img, (x, y))
    return orientations


#######################
# Gaussians averaging #
#######################


def create_1D_gaussian_kernel(sigma, size=0, epsilon=1e-6):
    """
    Create a 1D Gaussian kernel of a given sigma
    sigma: int
    epsilon: threshold to know if size is null (default is 1e-6)
    Return: 1D numpy array
    """
    if size < epsilon:
        # if no size is passed, define it with sigma
        size = 2 * np.ceil(3 * sigma) + 1
        size = int(size)
    x = np.linspace(-size / 2, size / 2, size)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


def convolve_2D_gaussian(g_img, sigma, size=0, mode="same"):
    """
    Convolve a grayscale image with a 2D Gaussian kernel
    g_img: float32 grayscale image
    sigma: float
    size: int size of kernel (default is 0)
    mode: str mode of convolution (default is "same")
    Return: float32 grayscale image
    """
    kernel = create_1D_gaussian_kernel(sigma, size)

    convolved_image_rows = fftconvolve(g_img, kernel.reshape(1, -1), mode=mode)

    # Convolution along columns
    convolved_image = fftconvolve(
        convolved_image_rows, kernel.reshape(-1, 1), mode=mode
    )

    return convolved_image


def compute_gaussian_mean(array, sigma):
    """
    Compute mean of a square array centered on its center weighted by a 2D Gaussian.
    Array possibly contains 2D vectors.
    array: float32 array, of shape (size, size, elt_dim) where size = 2*radius+1
    sigma: float, std of the Gaussian
    """
    # create the 1D Gaussian kernel of shape (size,) and sum 1
    size = array.shape[0]
    kernel1D = create_1D_gaussian_kernel(sigma, size)
    # create the 2D Gaussian kernel of shape (size, size) and sum 1
    kernel2D = np.dot(kernel1D.reshape(-1, 1), kernel1D.reshape(1, -1))
    # print(kernel2D.shape, kernel2D.sum())
    # compute the gaussian mean, eventually a vector
    if array.ndim > 2:
        g_mean = np.zeros((array.shape[2],), dtype=np.float32)
        for i in range(array.shape[2]):
            g_mean[i] = np.sum(array[:, :, i] * kernel2D)
    else:
        g_mean = np.sum(array * kernel2D)
    return g_mean


###########################
# Compute feature vectors #
###########################


def compute_angle(v1, v2, eps=1e-6):
    """
    Compute the angle between 2 vectors, in radians, in [0,2*pi].
    horizontal vector is always passed as second argument.
    Return 0 if one of the vectors is null.
    """
    norms = [np.linalg.norm(v1), np.linalg.norm(v2)]
    if norms[0] < eps or norms[1] < eps:
        return 0.0
    angle = np.arccos(np.dot(v1, v2) / (norms[0] * norms[1]))
    # decide between 2 possible angles with cross product
    if np.cross(v2, v1) < 0:
        angle = 2 * np.pi - angle
    return angle


def compute_horiz_angles(arrvects):
    """
    compute angle between each vector of the given array and the horizontal vector (0, 1) (in the image's frame)
    arrvects: numpy array of *non null* 2D vectors, of shape (height, width, 2) of float32 (recall vectors are the rows and not the columns)
    so arrvects[0,0,:] is the vector at the position (0,0) of the image
    return horiz_angles: numpy array of shape (height, width) of float32 with angles in radians
    """
    horiz_vect = np.array([0, 1], dtype=np.float32)

    def helper(vector):
        return compute_angle(vector, horiz_vect)

    horiz_angles = np.apply_along_axis(helper, -1, arrvects)

    return horiz_angles


def split_posneg_parts(array):
    """
    Compute array of positive and negative parts of a given array.
    array: numpy array, shape: (h,w)
    return pos_array, neg_array: numpy arrays, shape: (h,w)
    """
    pos_array = array * (array > 0)
    neg_array = array * (array < 0) * (-1)
    return pos_array, neg_array


def compute_features_overall_posneg(g_img, border_size=1):
    """
    Compute useful features for all pixels of a grayscale image.
    Return a list of pairs of features arrays, each pair containing the feature values and the feature orientations.
    All angular features are in degrees in [0, 360[.
    """
    # compute eigenvalues and eigenvectors of the Hessian matrix
    # recall objects in the avoided border are set to 0
    eigvals, eigvects, gradients = vh.compute_hessian_gradient_subimage(
        g_img, border_size
    )
    # compute gradients norms
    gradients_norms = np.linalg.norm(gradients, axis=2)

    # compute principal directions of the Hessian matrix
    principal_directions = np.zeros(eigvects.shape[:3], dtype=np.float32)
    for eigvect_id in range(2):
        principal_directions[:, :, eigvect_id] = compute_horiz_angles(
            eigvects[:, :, eigvect_id]
        )

    # compute orientations of the gradients in degrees
    orientations = compute_orientations(g_img, border_size)
    # convert and rescale angles in [0, 360[
    posdeg_orientations = convert_angles_to_pos_degrees(orientations)
    # same for principal directions (therefore they are angles)
    posdeg_principal_directions = np.zeros(
        principal_directions.shape[:3], dtype=np.float32
    )
    for prin_dir_id in range(2):
        posdeg_principal_directions[:, :, prin_dir_id] = convert_angles_to_pos_degrees(
            principal_directions[:, :, prin_dir_id]
        )

    # separate eigenvalues according to their sign
    # splitted_eigvals = split_eigenvalues(eigvals)
    # eigvals1pos, eigvals2pos, eigvals1neg, eigvals2neg = splitted_eigvals
    eigvals1pos, eigvals1neg = split_posneg_parts(eigvals[:, :, 0])
    eigvals2pos, eigvals2neg = split_posneg_parts(eigvals[:, :, 1])

    features = [
        posdeg_principal_directions,
        eigvals1pos,
        eigvals2pos,
        eigvals1neg,
        eigvals2neg,
        gradients_norms,
        posdeg_orientations,
    ]

    return features


def make_eigvals_positive(eigvals, eigvects):
    """
    Make eigenvalues positive by multiplying their corresponding eigenvectors by their signs.
    eigvals: numpy array of shape (h, w)
    eigvects: numpy array of shape (h, w, 2)
    return: signed_eigvects: numpy array of shape (h, w, 2)
    """
    signed_eigvects = eigvects * (eigvals > 0) + eigvects * (eigvals < 0) * (-1)
    return signed_eigvects


def compute_features_overall_abs(g_img, border_size=1):
    """
    Compute useful features for all pixels of a grayscale image.
    All angular features are in degrees in [0, 360[.
    Return signed principal directions, absolute value of eigenvalues, gradients norms and orientations.
    """
    # compute eigenvalues and eigenvectors of the Hessian matrix
    # recall objects in the avoided border are set to 0
    eigvals, eigvects, gradients = vh.compute_hessian_gradient_subimage(
        g_img, border_size
    )

    # compute eigenvalues absolute values
    abs_eigvals = np.abs(eigvals)
    # compute gradients norms
    gradients_norms = np.linalg.norm(gradients, axis=2)

    # Make eigenvalues positive by multiplying their corresponding eigenvectors by their signs.
    # ie 180° rotation
    signed_eigvects = np.zeros_like(eigvects, dtype=np.float32)
    for eigval_id in range(2):
        signed_eigvects[:, :, eigval_id] = (
            eigvects[:, :, eigval_id]
            * ((eigvals[:, :, eigval_id] > 0)[:, :, np.newaxis])
            + eigvects[:, :, eigval_id]
            * ((eigvals[:, :, eigval_id] < 0) * (-1))[:, :, np.newaxis]
        )

    # compute principal directions of the Hessian matrix as angles with horizontal vector
    signed_principal_directions = np.zeros(eigvects.shape[:3], dtype=np.float32)
    for eigvect_id in range(2):
        signed_principal_directions[:, :, eigvect_id] = compute_horiz_angles(
            signed_eigvects[:, :, eigvect_id]
        )
    # compute orientations of the gradients in degrees
    orientations = compute_orientations(g_img, border_size)

    # convert and rescale angles in [0, 360[
    posdeg_orientations = convert_angles_to_pos_degrees(orientations)
    # same for principal directions (therefore they are angles)
    posdeg_signed_principal_directions = np.zeros(
        signed_principal_directions.shape[:3], dtype=np.float32
    )
    for prin_dir_id in range(2):
        posdeg_signed_principal_directions[
            :, :, prin_dir_id
        ] = convert_angles_to_pos_degrees(
            signed_principal_directions[:, :, prin_dir_id]
        )

    features = [
        posdeg_signed_principal_directions,
        abs_eigvals,
        gradients_norms,
        posdeg_orientations,
    ]

    return features


######################
# Compute histograms #
######################


def rescale_value(values, kp_value, epsilon=1e-6):
    """
    Rescale vector values by keypoint value with broadcast.
    Vector value must be positive.
    If kp_value is null, divide by 1 instead.
    epsilon: float to avoid division by 0 (default is 1e-6)
    """
    if kp_value < epsilon:
        print("Warning: keypoint value is null, rescale by 1 instead")
        kp_value = epsilon  # could use another value such as 0.1 or 0.01 ? to better convey the fact that the value is null at that point
    return values / kp_value


def rotate_orientations(orientations, kp_orientation):
    """
    Rotate orientations with kp orientation in trigo order with broadcast.
    """
    rotated_orientations = (orientations - kp_orientation) % 360.0
    return rotated_orientations


def compute_vector_histogram(
    rescaled_values,
    rotated_orientations,
    kp_position,
    nb_bins=3,
    bin_radius=2,
    delta_angle=5.0,
    sigma=0,
):
    """
    Discretize neighborhood of keypoint into nb_bins bins, each of size 2*bin_radius+1, centered on keypoint.
    For each neighborhood, compute the histogram of the vectors orientations, weighted by vector values and distance to keypoint.
    Vector value must be positive.
    rescaled_values, rotated_orientations: numpy arrays of shape (height, width) (same as the image)
    kp_position: (x, y) int pixel position of keypoint in the image frame
    nb_bins: int odd number of bins of the neighborhood (default is 3)
    delta_angle: float size of the angular bins in degrees (default is 5)
    """
    # if sigma is null, define it with neighborhood_width
    neighborhood_width = nb_bins * (2 * bin_radius + 1)
    if sigma == 0:
        sigma = neighborhood_width / 2
    # compute the gaussian weight of the vectors
    g_weight = np.exp(-1 / (2 * sigma**2))

    # initialize the angular histograms
    nb_angular_bins = int(360 / delta_angle) + 1
    histograms = np.zeros((nb_bins, nb_bins, nb_angular_bins), dtype=np.float32)

    # loop over all pixels and add its contribution to the right angular histogram
    for bin_j in range(nb_bins):
        for bin_i in range(nb_bins):
            # compute the center of the neighborhood in the frame of the image
            x_center = kp_position[0] + (bin_j - nb_bins // 2) * (2 * bin_radius + 1)
            y_center = kp_position[1] + (bin_i - nb_bins // 2) * (2 * bin_radius + 1)
            # loop over all pixels of the neighborhood in the frame of the image
            for j in range(y_center - bin_radius, y_center + bin_radius + 1):
                for i in range(x_center - bin_radius, x_center + bin_radius + 1):
                    # compute the angle of the vector
                    angle = rotated_orientations[j, i]
                    # compute the gaussian weight of the vector
                    weight = g_weight * np.exp(
                        -((i - kp_position[0]) ** 2 + (j - kp_position[1]) ** 2)
                        / (2 * sigma**2)
                    )
                    # compute the bin of the angle
                    bin_angle = np.round(angle / delta_angle)
                    # add the weighted vector to the right histogram
                    histograms[bin_j, bin_i, bin_angle] += (
                        weight * rescaled_values[j, i]
                    )

            # normalize the histogram, now in percentage
            histograms[bin_j, bin_i, :] /= np.sum(histograms[bin_j, bin_i, :])
            histograms[bin_j, bin_i, :] *= 100.0

    return histograms


def rotate_point(x, y, angle, center_x, center_y):
    """
    Rotate a point counterclockwise by a given angle around a given origin, even if the vertical axis points downward.
    x, y: coordinates of the point
    angle: rotation angle in degrees
    center_x, center_y: coordinates of the rotation center
    return: new coordinates of the point
    """
    # Convert angle from degrees to radians
    angle = np.deg2rad(angle)

    # Translate point back to origin
    x -= center_x
    y -= center_y

    # Rotate point
    new_x = x * np.cos(angle) - y * np.sin(angle)
    new_y = x * np.sin(angle) + y * np.cos(angle)

    # Translate point back
    new_x += center_x
    new_y += center_y

    return new_x, new_y


def rotate_point_pixel(row, col, angle, center_row, center_col):
    """
    Rotate a point clockwise by a given angle around a given origin.
    row, col: coordinates of the point (in pixel coordinates)
    angle: rotation angle in degrees
    center_row, center_col: coordinates of the rotation center (in pixel coordinates)
    return: new coordinates of the point (in pixel coordinates)
    """
    # Convert angle from degrees to radians
    angle = np.deg2rad(angle)

    # Invert the row coordinate (because the y-axis points downward)
    row = -row
    center_row = -center_row

    # Translate point back to origin
    x = col - center_col
    y = row - center_row

    # Rotate point
    new_x = x * np.cos(angle) - y * np.sin(angle)
    new_y = x * np.sin(angle) + y * np.cos(angle)

    # Translate point back
    new_x += center_col
    new_y += center_row

    # Invert the new row coordinate back
    new_y = -new_y

    # round to int
    new_y = np.round(new_y).astype(np.int32)
    new_x = np.round(new_x).astype(np.int32)

    return new_y, new_x


# def compute_rotated_vector_histogram(
#     rescaled_values,
#     rotated_orientations,
#     kp_position,
#     kp_gradient_orientation,
#     nb_bins=3,
#     bin_radius=2,
#     delta_angle=5.0,
#     sigma=0,
# ):
#     """
#     Discretize neighborhood of keypoint into nb_bins bins, each of size 2*bin_radius+1, centered on keypoint.
#     Rotate neighborhood in the orientation of the keypoint
#     For each spatial bin, compute the histogram of the vectors orientations, weighted by vector values and distance to keypoint.
#     Vector value must be positive.
#     rescaled_values, rotated_orientations: numpy arrays of shape (height, width) (same as the image)
#     kp_position: (x, y) int pixel position of keypoint in the image frame
#     nb_bins: int odd number of bins of the neighborhood (default is 3)
#     delta_angle: float size of the angular bins in degrees (default is 5)
#     """
#     # if sigma is null, define it with neighborhood_width
#     neighborhood_width = nb_bins * (2 * bin_radius + 1)
#     if sigma == 0:
#         sigma = neighborhood_width / 2
#     # compute the gaussian weight of the vectors
#     g_weight = np.exp(-1 / (2 * sigma**2))

#     # initialize the angular histograms
#     nb_angular_bins = int(360 / delta_angle) + 1
#     histograms = np.zeros((nb_bins, nb_bins, nb_angular_bins), dtype=np.float32)

#     # rotation is with angle +kp_gradient_orientation counterclockwise
#     # loop over all pixels and add its contribution to the right angular histogram
#     # recall i is the row index and j the column index
#     for bin_i in range(nb_bins):
#         for bin_j in range(nb_bins):
#             # compute the center of the neighborhood in the frame of the image
#             x_center = kp_position[0] + (bin_j - nb_bins // 2) * (2 * bin_radius + 1)
#             y_center = kp_position[1] + (bin_i - nb_bins // 2) * (2 * bin_radius + 1)

#             # loop over all pixels of the neighborhood
#             # still in the frame of the image
#             for i in range(y_center - bin_radius, y_center + bin_radius + 1):
#                 for j in range(x_center - bin_radius, x_center + bin_radius + 1):
#                     # compute rotated pixel position in the image frame
#                     # because rotate_point_pixel rotate a positive angle clockwise,
#                     # we need to rotate with -kp_gradient_orientation
#                     i_rot_img, j_rot_img = rotate_point_pixel(
#                         i, j, -kp_gradient_orientation, kp_position[1], kp_position[0]
#                     )

#                     # compute the angle of the vector in the frame of the image
#                     angle = rotated_orientations[i_rot_img, j_rot_img]
#                     # compute the contribution of the vector in the frame of the image
#                     contribution = rescaled_values[i_rot_img, j_rot_img]

#                     # compute the gaussian weight of the vector with the distance in the image frame
#                     weight = g_weight * np.exp(
#                         -(
#                             (i_rot_img - kp_position[1]) ** 2
#                             + (j_rot_img - kp_position[0]) ** 2
#                         )
#                         / (2 * sigma**2)
#                     )
#                     # compute the bin of the angle
#                     bin_angle = np.round(angle / delta_angle).astype(np.int32)

#                     # add the weighted vector to the right histogram
#                     histograms[bin_i, bin_j, bin_angle] += weight * contribution

#             # normalize the histogram, now in percentage
#             histograms[bin_i, bin_j, :] /= np.sum(histograms[bin_i, bin_j, :])
#             histograms[bin_i, bin_j, :] *= 100.0

#     return histograms


def compute_rotated_vector_histogram(
    rescaled_values,
    rotated_orientations,
    kp_position,
    kp_gradient_orientation,
    nb_bins=3,
    bin_radius=2,
    delta_angle=5.0,
    sigma=0,
    normalization_mode="global",
):
    """
    Discretize neighborhood of keypoint into nb_bins bins, each of size 2*bin_radius+1, centered on keypoint.
    Rotate neighborhood in the orientation of the keypoint
    For each spatial bin, compute the histogram of the vectors orientations, weighted by vector values and distance to keypoint.
    Vector value must be positive.
    rescaled_values, rotated_orientations: numpy arrays of shape (height, width) (same as the image)
    kp_position: (x, y) int pixel position of keypoint in the image frame
    nb_bins: int odd number of bins of the neighborhood (default is 3)
    delta_angle: float size of the angular bins in degrees (default is 5)
    sigma: float std of the gaussian weight (default is 0, then automatically defined with neighborhood_width)
    normalization_mode: str mode of normalization (default is "global"), dictates how to normalize the histogram
    if "global", normalize by overall sum of contributions in all the histograms,
    if "local", normalize each histogram only by the sum of its contributions
    """
    # if sigma is null, define it with neighborhood_width
    neighborhood_width = nb_bins * (2 * bin_radius + 1)
    if sigma == 0:
        sigma = neighborhood_width / 2
    # compute the gaussian weight of the vectors
    g_weight = np.exp(-1 / (2 * sigma**2))

    # initialize the angular histograms
    nb_angular_bins = int(360 / delta_angle) + 1
    histograms = np.zeros((nb_bins, nb_bins, nb_angular_bins), dtype=np.float32)

    # rotation is with angle +kp_gradient_orientation counterclockwise
    # loop over all pixels and add its contribution to the right angular histogram
    # recall i is the row index and j the column index
    for bin_i in range(nb_bins):
        for bin_j in range(nb_bins):
            # compute the center of the neighborhood in the frame of the image
            x_center = kp_position[0] + (bin_j - nb_bins // 2) * (2 * bin_radius + 1)
            y_center = kp_position[1] + (bin_i - nb_bins // 2) * (2 * bin_radius + 1)

            # loop over all pixels of the neighborhood
            # still in the frame of the image
            for i in range(y_center - bin_radius, y_center + bin_radius + 1):
                for j in range(x_center - bin_radius, x_center + bin_radius + 1):
                    # compute rotated pixel position in the image frame
                    # because rotate_point_pixel rotate a positive angle clockwise,
                    # we need to rotate with -kp_gradient_orientation
                    i_rot_img, j_rot_img = rotate_point_pixel(
                        i, j, -kp_gradient_orientation, kp_position[1], kp_position[0]
                    )

                    # compute the angle of the vector in the frame of the image
                    angle = rotated_orientations[i_rot_img, j_rot_img]
                    # compute the contribution of the vector in the frame of the image
                    contribution = rescaled_values[i_rot_img, j_rot_img]

                    # compute the gaussian weight of the vector with the distance in the image frame
                    weight = g_weight * np.exp(
                        -(
                            (i_rot_img - kp_position[1]) ** 2
                            + (j_rot_img - kp_position[0]) ** 2
                        )
                        / (2 * sigma**2)
                    )
                    # compute the bin of the angle
                    bin_angle = np.round(angle / delta_angle).astype(np.int32)

                    # add the weighted vector to the right histogram
                    histograms[bin_i, bin_j, bin_angle] += weight * contribution

                    if normalization_mode == "local":
                        histograms[bin_i, bin_j, :] *= 100.0 / np.sum(
                            histograms[bin_i, bin_j, :]
                        )

    if normalization_mode == "global":
        histograms *= 100.0 / np.sum(histograms)

    return histograms


######################
# Compute descriptor #
######################


def compute_descriptor_histograms_posneg(
    overall_features_posneg,
    kp_position,
    nb_bins=3,
    bin_radius=2,
    delta_angle=5.0,
    sigma=0,
):
    """
    Compute the histograms for the descriptor of a keypoint
    overall_features_posneg: list of features arrays of all pixels of the image
    kp_position: (x, y) int pixel position of keypoint in the image frame
    return descriptor_histograms: list of 3 histograms, each of shape (nb_bins, nb_bins, nb_angular_bins)
    """
    x_kp, y_kp = kp_position

    # unpack the features
    (
        posdeg_principal_directions,
        eigvals1pos,
        eigvals2pos,
        eigvals1neg,
        eigvals2neg,
        gradients_norms,
        posdeg_orientations,
    ) = overall_features_posneg

    # compute absolute value of keypoint eigenvalue for rescale
    kp_abs_eigval1 = eigvals1pos[y_kp, x_kp] + eigvals1neg[y_kp, x_kp]
    kp_abs_eigval2 = eigvals2pos[y_kp, x_kp] + eigvals2neg[y_kp, x_kp]

    # rescale vectors values by keypoint value
    rescaled_eigvals1pos = rescale_value(eigvals1pos, kp_abs_eigval1)
    rescaled_eigvals2pos = rescale_value(eigvals2pos, kp_abs_eigval2)
    rescaled_eigvals1neg = rescale_value(eigvals1neg, kp_abs_eigval1)
    rescaled_eigvals2neg = rescale_value(eigvals2neg, kp_abs_eigval2)

    # rotate vectors orientations with mean of kp orientation in trigo order
    # therefore it is a angle of reference for the keypoint that does not depend on the
    # order of the eigenvalues
    kp_prin_dirs = posdeg_principal_directions[y_kp, x_kp]
    mean_kp_prin_dir = 0.5 * (kp_prin_dirs[0] + kp_prin_dirs[1])
    rotated_prin_dirs = [
        rotate_orientations(
            posdeg_principal_directions[:, :, i],
            mean_kp_prin_dir,
        )
        for i in range(2)
    ]

    kp_orientation = posdeg_orientations[y_kp, x_kp]
    rotated_orientations = rotate_orientations(
        posdeg_orientations,
        kp_orientation,
    )

    # compute 1st positive eigenvalues histogram
    eig1pos_hist = compute_vector_histogram(
        rescaled_eigvals1pos,
        rotated_prin_dirs[0],
        kp_position,
        nb_bins,
        bin_radius,
        delta_angle,
        sigma,
    )

    # compute 2nd positive eigenvalues histogram
    eig2pos_hist = compute_vector_histogram(
        rescaled_eigvals2pos,
        rotated_prin_dirs[1],
        kp_position,
        nb_bins,
        bin_radius,
        delta_angle,
        sigma,
    )

    # compute 1st negative eigenvalues histogram
    eig1neg_hist = compute_vector_histogram(
        rescaled_eigvals1neg,
        rotated_prin_dirs[0],
        kp_position,
        nb_bins,
        bin_radius,
        delta_angle,
        sigma,
    )

    # compute 2nd negative eigenvalues histogram
    eig2neg_hist = compute_vector_histogram(
        rescaled_eigvals2neg,
        rotated_prin_dirs[1],
        kp_position,
        nb_bins,
        bin_radius,
        delta_angle,
        sigma,
    )

    # compute gradients histogram
    grad_hist = compute_vector_histogram(
        gradients_norms,
        rotated_orientations,
        kp_position,
        nb_bins,
        bin_radius,
        delta_angle,
        sigma,
    )

    # Add histograms of same sign eigenvalues
    eigpos_hist = eig1pos_hist + eig2pos_hist
    eigneg_hist = eig1neg_hist + eig2neg_hist

    # stack all histograms
    descriptor_histograms = [eigpos_hist, eigneg_hist, grad_hist]

    return descriptor_histograms


def compute_descriptor_histograms_1_2(
    overall_features_1_2, kp_position, nb_bins=3, bin_radius=2, delta_angle=5.0, sigma=0
):
    """
    Compute the histograms for the descriptor of a keypoint
    overall_features_posneg: list of features arrays of all pixels of the image
    kp_position: (x, y) int pixel position of keypoint in the image frame
    return descriptor_histograms: list of 3 histograms, each of shape (nb_bins, nb_bins, nb_angular_bins)
    1st histogram: first eigenvalues (highest signed value)
    2nd histogram: second eigenvalues
    3rd histogram: gradients
    """
    x_kp, y_kp = kp_position
    # unpack the features
    (
        posdeg_principal_directions,
        abs_eigvals,
        gradients_norms,
        posdeg_orientations,
    ) = overall_features_1_2

    # compute absolute value of keypoint eigenvalue for rescale
    kp_abs_eigvals = abs_eigvals[y_kp, x_kp]

    # rescale vectors values by keypoint value
    rescaled_eigvals = np.zeros_like(abs_eigvals, dtype=np.float32)
    for eigval_id in range(2):
        rescaled_eigvals[:, :, eigval_id] = rescale_value(
            abs_eigvals[:, :, eigval_id], kp_abs_eigvals[eigval_id]
        )

    # rotate vectors orientations
    # principal directions
    kp_prin_dirs = posdeg_principal_directions[y_kp, x_kp]
    rotated_prin_dirs = np.zeros_like(posdeg_principal_directions, dtype=np.float32)
    for eigval_id in range(2):
        rotated_prin_dirs[:, :, eigval_id] = rotate_orientations(
            posdeg_principal_directions[:, :, eigval_id],
            kp_prin_dirs[eigval_id],
        )
    # gradients orientations
    kp_orientation = posdeg_orientations[y_kp, x_kp]
    rotated_orientations = rotate_orientations(
        posdeg_orientations,
        kp_orientation,
    )

    # compute the 2 eigenvalues histograms
    eigvals_hists = [
        compute_vector_histogram(
            rescaled_eigvals[:, :, eigval_id],
            rotated_prin_dirs[:, :, eigval_id],
            kp_position,
            nb_bins,
            bin_radius,
            delta_angle,
            sigma,
        )
        for eigval_id in range(2)
    ]

    # compute gradients histogram
    grad_hist = compute_vector_histogram(
        gradients_norms,
        rotated_orientations,
        kp_position,
        nb_bins,
        bin_radius,
        delta_angle,
        sigma,
    )

    # stack all histograms
    descriptor_histograms = [eigvals_hists[0], eigvals_hists[1], grad_hist]

    return descriptor_histograms


# def compute_descriptor_histograms_1_2_rotated(
#     overall_features_1_2, kp_position, nb_bins=3, bin_radius=2, delta_angle=5.0, sigma=0
# ):
#     """
#     Compute the histograms for the descriptor of a keypoint, with the rotated neighborhood.
#     overall_features_posneg: list of features arrays of all pixels of the image
#     kp_position: (x, y) int pixel position of keypoint in the image frame
#     return descriptor_histograms: list of 3 histograms, each of shape (nb_bins, nb_bins, nb_angular_bins)
#     1st histogram: first eigenvalues (highest signed value)
#     2nd histogram: second eigenvalues
#     3rd histogram: gradients
#     """
#     x_kp, y_kp = kp_position
#     # unpack the features
#     (
#         posdeg_principal_directions,
#         abs_eigvals,
#         gradients_norms,
#         posdeg_orientations,
#     ) = overall_features_1_2

#     # compute absolute value of keypoint eigenvalue for rescale
#     kp_abs_eigvals = abs_eigvals[y_kp, x_kp]

#     # rescale vectors values by keypoint value
#     rescaled_eigvals = np.zeros_like(abs_eigvals, dtype=np.float32)
#     for eigval_id in range(2):
#         rescaled_eigvals[:, :, eigval_id] = rescale_value(
#             abs_eigvals[:, :, eigval_id], kp_abs_eigvals[eigval_id]
#         )

#     # rotate vectors orientations
#     # principal directions
#     kp_prin_dirs = posdeg_principal_directions[y_kp, x_kp]
#     rotated_prin_dirs = np.zeros_like(posdeg_principal_directions, dtype=np.float32)
#     for eigval_id in range(2):
#         rotated_prin_dirs[:, :, eigval_id] = rotate_orientations(
#             posdeg_principal_directions[:, :, eigval_id],
#             kp_prin_dirs[eigval_id],
#         )
#     # gradients orientations
#     kp_orientation = posdeg_orientations[y_kp, x_kp]
#     rotated_orientations = rotate_orientations(
#         posdeg_orientations,
#         kp_orientation,
#     )

#     # compute the 2 eigenvalues histograms
#     eigvals_hists = [
#         compute_rotated_vector_histogram(
#             rescaled_eigvals[:, :, eigval_id],
#             rotated_prin_dirs[:, :, eigval_id],
#             kp_position,
#             kp_orientation,
#             nb_bins,
#             bin_radius,
#             delta_angle,
#             sigma,
#         )
#         for eigval_id in range(2)
#     ]

#     # compute gradients histogram
#     grad_hist = compute_rotated_vector_histogram(
#         gradients_norms,
#         rotated_orientations,
#         kp_position,
#         kp_orientation,
#         nb_bins,
#         bin_radius,
#         delta_angle,
#         sigma,
#     )

#     # stack all histograms
#     descriptor_histograms = [eigvals_hists[0], eigvals_hists[1], grad_hist]

#     return descriptor_histograms


def compute_descriptor_histograms_1_2_rotated(
    overall_features_1_2,
    kp_position,
    nb_bins=3,
    bin_radius=2,
    delta_angle=5.0,
    sigma=0,
    normalization_mode="global",
):
    """
    Compute the histograms for the descriptor of a keypoint, with the rotated neighborhood.
    overall_features_posneg: list of features arrays of all pixels of the image
    kp_position: (x, y) int pixel position of keypoint in the image frame
    normalization_mode: str mode of normalization (default is "global"), dictates how to normalize the histogram
    return descriptor_histograms: list of 3 histograms, each of shape (nb_bins, nb_bins, nb_angular_bins)
    1st histogram: first eigenvalues (highest signed value)
    2nd histogram: second eigenvalues
    3rd histogram: gradients
    """
    x_kp, y_kp = kp_position
    # unpack the features
    (
        posdeg_principal_directions,
        abs_eigvals,
        gradients_norms,
        posdeg_orientations,
    ) = overall_features_1_2

    # compute absolute value of keypoint eigenvalue for rescale
    kp_abs_eigvals = abs_eigvals[y_kp, x_kp]

    # rescale vectors values by keypoint value
    rescaled_eigvals = np.zeros_like(abs_eigvals, dtype=np.float32)
    for eigval_id in range(2):
        rescaled_eigvals[:, :, eigval_id] = rescale_value(
            abs_eigvals[:, :, eigval_id], kp_abs_eigvals[eigval_id]
        )

    # rotate vectors orientations
    # principal directions
    kp_prin_dirs = posdeg_principal_directions[y_kp, x_kp]
    rotated_prin_dirs = np.zeros_like(posdeg_principal_directions, dtype=np.float32)
    for eigval_id in range(2):
        rotated_prin_dirs[:, :, eigval_id] = rotate_orientations(
            posdeg_principal_directions[:, :, eigval_id],
            kp_prin_dirs[eigval_id],
        )
    # gradients orientations
    kp_orientation = posdeg_orientations[y_kp, x_kp]
    rotated_orientations = rotate_orientations(
        posdeg_orientations,
        kp_orientation,
    )

    # compute the 2 eigenvalues histograms
    eigvals_hists = [
        compute_rotated_vector_histogram(
            rescaled_eigvals[:, :, eigval_id],
            rotated_prin_dirs[:, :, eigval_id],
            kp_position,
            kp_orientation,
            nb_bins,
            bin_radius,
            delta_angle,
            sigma,
            normalization_mode,
        )
        for eigval_id in range(2)
    ]

    # compute gradients histogram
    grad_hist = compute_rotated_vector_histogram(
        gradients_norms,
        rotated_orientations,
        kp_position,
        kp_orientation,
        nb_bins,
        bin_radius,
        delta_angle,
        sigma,
        normalization_mode,
    )

    # stack all histograms
    descriptor_histograms = [eigvals_hists[0], eigvals_hists[1], grad_hist]

    return descriptor_histograms


#######################
# Descriptor Matching #
#######################


def flatten_descriptor(descriptor_histograms):
    """
    Flatten the descriptor histograms into a 1D vector.
    descriptor_histograms: list of 3 histograms, each of shape (nb_bins, nb_bins, nb_angular_bins)
    return descriptor: 1D numpy array, length = nb_bins * nb_bins * nb_angular_bins * 3
    """
    descriptor = np.concatenate(
        [
            descriptor_histograms[0].flatten(),
            descriptor_histograms[1].flatten(),
            descriptor_histograms[2].flatten(),
        ]
    )
    return descriptor


def compute_descriptor_distance(descriptor1, descriptor2):
    """
    Compute the euclidean distance between 2 descriptors.
    descriptor1, descriptor2: 1D numpy arrays
    return distance: float
    """
    distance = np.linalg.norm(descriptor1 - descriptor2)
    return distance
import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, fftconvolve
import scipy.ndimage as ndimage

# add the path to the visualize_hessian folder
sys.path.append(os.path.join("..", "visualize_hessian"))
# import visualize_hessian.visu_hessian
import visu_hessian as vh

# return to the root directory
sys.path.append(os.path.join(".."))

################
# Orientations #
################


def compute_orientation(g_img, position, epsilon=1e-6):
    """
    Compute the orientation of the keypoint with Lowe's formula.
    g_img: float32 grayscale image
    position: (x, y) int pixel position of point
    epsilon: float to avoid division by 0 (default is 1e-6)
    return orientation: float32 orientation of the keypoint in degrees, within [0, 360[
    """
    x, y = position
    denominator = g_img[y, x + 1] - g_img[y, x - 1]
    # if denominator is too small, add or subtract epsilon to avoid division by 0 (convention taken)
    if np.abs(denominator) < epsilon:
        if denominator >= 0:
            denominator += epsilon
        else:
            denominator -= epsilon
    slope = (g_img[y + 1, x] - g_img[y - 1, x]) / denominator
    orientation = np.arctan(slope)

    return orientation


def convert_angles_to_pos_degrees(angles):
    """
    Convert array of angles in radians to positive degrees in [0, 360[
    """
    # translate the angles in [0, 2pi[
    posdeg_angles = angles % (2 * np.pi)
    # translate the orientation in [0, 360[
    posdeg_angles = posdeg_angles * 180 / np.pi
    return posdeg_angles


def compute_orientations(g_img, border_size=1):
    """
    Compute the orientation of all pixels of a grayscale image within a border
    g_img: float32 grayscale image
    border_size: int size of the border to ignore (default is 1)
    return orientations: float32 array of shape (height, width) of the orientations in radians
    """
    orientations = np.zeros_like(g_img, dtype=np.float32)
    for y in range(border_size, g_img.shape[0] - border_size):
        for x in range(border_size, g_img.shape[1] - border_size):
            orientations[y, x] = compute_orientation(g_img, (x, y))
    return orientations


#######################
# Gaussians averaging #
#######################


def create_1D_gaussian_kernel(sigma, size=0, epsilon=1e-6):
    """
    Create a 1D Gaussian kernel of a given sigma
    sigma: int
    epsilon: threshold to know if size is null (default is 1e-6)
    Return: 1D numpy array
    """
    if size < epsilon:
        # if no size is passed, define it with sigma
        size = 2 * np.ceil(3 * sigma) + 1
        size = int(size)
    x = np.linspace(-size / 2, size / 2, size)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


def convolve_2D_gaussian(g_img, sigma, size=0, mode="same"):
    """
    Convolve a grayscale image with a 2D Gaussian kernel
    g_img: float32 grayscale image
    sigma: float
    size: int size of kernel (default is 0)
    mode: str mode of convolution (default is "same")
    Return: float32 grayscale image
    """
    kernel = create_1D_gaussian_kernel(sigma, size)

    convolved_image_rows = fftconvolve(g_img, kernel.reshape(1, -1), mode=mode)

    # Convolution along columns
    convolved_image = fftconvolve(
        convolved_image_rows, kernel.reshape(-1, 1), mode=mode
    )

    return convolved_image


def compute_gaussian_mean(array, sigma):
    """
    Compute mean of a square array centered on its center weighted by a 2D Gaussian.
    Array possibly contains 2D vectors.
    array: float32 array, of shape (size, size, elt_dim) where size = 2*radius+1
    sigma: float, std of the Gaussian
    """
    # create the 1D Gaussian kernel of shape (size,) and sum 1
    size = array.shape[0]
    kernel1D = create_1D_gaussian_kernel(sigma, size)
    # create the 2D Gaussian kernel of shape (size, size) and sum 1
    kernel2D = np.dot(kernel1D.reshape(-1, 1), kernel1D.reshape(1, -1))
    # print(kernel2D.shape, kernel2D.sum())
    # compute the gaussian mean, eventually a vector
    if array.ndim > 2:
        g_mean = np.zeros((array.shape[2],), dtype=np.float32)
        for i in range(array.shape[2]):
            g_mean[i] = np.sum(array[:, :, i] * kernel2D)
    else:
        g_mean = np.sum(array * kernel2D)
    return g_mean


###########################
# Compute feature vectors #
###########################


# def compute_angle(v1, v2, eps=1e-6):
#     """
#     Compute the angle between 2 vectors, in radians.
#     horizontal vector is always passed as second argument.
#     Return 0 if one of the vectors is null.
#     """
#     norms = [np.linalg.norm(v1), np.linalg.norm(v2)]
#     if norms[0] < eps or norms[1] < eps:
#         return 0.0
#     angle = np.arccos(np.dot(v1, v2) / (norms[0] * norms[1]))
#     return angle


def compute_angle(v1, v2, eps=1e-6):
    """
    Compute the angle between 2 vectors, in radians, in [0,2*pi].
    horizontal vector is always passed as second argument.
    Return 0 if one of the vectors is null.
    """
    norms = [np.linalg.norm(v1), np.linalg.norm(v2)]
    if norms[0] < eps or norms[1] < eps:
        return 0.0
    angle = np.arccos(np.dot(v1, v2) / (norms[0] * norms[1]))
    # decide between 2 possible angles with cross product
    if np.cross(v2, v1) < 0:
        angle = 2 * np.pi - angle
    return angle


def compute_horiz_angles(arrvects):
    """
    compute angle between each vector of the given array and the horizontal vector (0, 1) (in the image's frame)
    arrvects: numpy array of *non null* 2D vectors, of shape (height, width, 2) of float32 (recall vectors are the rows and not the columns)
    so arrvects[0,0,:] is the vector at the position (0,0) of the image
    return horiz_angles: numpy array of shape (height, width) of float32 with angles in radians
    """
    horiz_vect = np.array([0, 1], dtype=np.float32)

    def helper(vector):
        return compute_angle(vector, horiz_vect)

    horiz_angles = np.apply_along_axis(helper, -1, arrvects)

    return horiz_angles


# def split_eigenvalues(eigvals):
#     # Compute for each array of eigenvalue, the arrays of their positive and negative parts
#     eigvals1pos = eigvals[:, :, 0] * (eigvals[:, :, 0] > 0)
#     eigvals1neg = eigvals[:, :, 0] * (eigvals[:, :, 0] < 0) * (-1)
#     eigvals2pos = eigvals[:, :, 1] * (eigvals[:, :, 1] > 0)
#     eigvals2neg = eigvals[:, :, 1] * (eigvals[:, :, 1] < 0) * (-1)

#     # return list of features in specific order
#     return [eigvals1pos, eigvals2pos, eigvals1neg, eigvals2neg]


def split_posneg_parts(array):
    """
    Compute array of positive and negative parts of a given array.
    array: numpy array, shape: (h,w)
    return pos_array, neg_array: numpy arrays, shape: (h,w)
    """
    pos_array = array * (array > 0)
    neg_array = array * (array < 0) * (-1)
    return pos_array, neg_array


def compute_features_overall(g_img, border_size=1):
    """
    Compute useful features for all pixels of a grayscale image.
    Return a list of pairs of features arrays, each pair containing the feature values and the feature orientations.
    All angular features are in degrees in [0, 360[.
    """
    # compute eigenvalues and eigenvectors of the Hessian matrix
    # recall objects in the avoided border are set to 0
    eigvals, eigvects, gradients = vh.compute_hessian_gradient_subimage(
        g_img, border_size
    )
    # compute gradients norms
    gradients_norms = np.linalg.norm(gradients, axis=2)

    # compute principal directions of the Hessian matrix
    principal_directions = np.zeros(eigvects.shape[:3], dtype=np.float32)
    for eigvect_id in range(2):
        principal_directions[:, :, eigvect_id] = compute_horiz_angles(
            eigvects[:, :, eigvect_id]
        )

    # compute orientations of the gradients in degrees
    orientations = compute_orientations(g_img, border_size)
    # convert and rescale angles in [0, 360[
    posdeg_orientations = convert_angles_to_pos_degrees(orientations)
    # same for principal directions (therefore they are angles)
    posdeg_principal_directions = np.zeros(
        principal_directions.shape[:3], dtype=np.float32
    )
    for prin_dir_id in range(2):
        posdeg_principal_directions[:, :, prin_dir_id] = convert_angles_to_pos_degrees(
            principal_directions[:, :, prin_dir_id]
        )

    # separate eigenvalues according to their sign
    # splitted_eigvals = split_eigenvalues(eigvals)
    # eigvals1pos, eigvals2pos, eigvals1neg, eigvals2neg = splitted_eigvals
    eigvals1pos, eigvals1neg = split_posneg_parts(eigvals[:, :, 0])
    eigvals2pos, eigvals2neg = split_posneg_parts(eigvals[:, :, 1])

    features = [
        posdeg_principal_directions,
        eigvals1pos,
        eigvals2pos,
        eigvals1neg,
        eigvals2neg,
        gradients_norms,
        posdeg_orientations,
    ]

    return features


def make_eigvals_positive(eigvals, eigvects):
    """
    Make eigenvalues positive by multiplying their corresponding eigenvectors by their signs.
    eigvals: numpy array of shape (h, w)
    eigvects: numpy array of shape (h, w, 2)
    return: signed_eigvects: numpy array of shape (h, w, 2)
    """
    signed_eigvects = eigvects * (eigvals > 0) + eigvects * (eigvals < 0) * (-1)
    return signed_eigvects


def compute_features_overall2(g_img, border_size=1):
    """
    Compute useful features for all pixels of a grayscale image.
    All angular features are in degrees in [0, 360[.
    Return signed principal directions, absolute value of eigenvalues, gradients norms and orientations.
    """
    # compute eigenvalues and eigenvectors of the Hessian matrix
    # recall objects in the avoided border are set to 0
    eigvals, eigvects, gradients = vh.compute_hessian_gradient_subimage(
        g_img, border_size
    )

    # compute eigenvalues absolute values
    abs_eigvals = np.abs(eigvals)
    # compute gradients norms
    gradients_norms = np.linalg.norm(gradients, axis=2)

    # Make eigenvalues positive by multiplying their corresponding eigenvectors by their signs.
    # ie 180° rotation
    signed_eigvects = np.zeros_like(eigvects, dtype=np.float32)
    for eigval_id in range(2):
        signed_eigvects[:, :, eigval_id] = (
            eigvects[:, :, eigval_id]
            * ((eigvals[:, :, eigval_id] > 0)[:, :, np.newaxis])
            + eigvects[:, :, eigval_id]
            * ((eigvals[:, :, eigval_id] < 0) * (-1))[:, :, np.newaxis]
        )

    # compute principal directions of the Hessian matrix as angles with horizontal vector
    signed_principal_directions = np.zeros(eigvects.shape[:3], dtype=np.float32)
    for eigvect_id in range(2):
        signed_principal_directions[:, :, eigvect_id] = compute_horiz_angles(
            signed_eigvects[:, :, eigvect_id]
        )
    # compute orientations of the gradients in degrees
    orientations = compute_orientations(g_img, border_size)

    # convert and rescale angles in [0, 360[
    posdeg_orientations = convert_angles_to_pos_degrees(orientations)
    # same for principal directions (therefore they are angles)
    posdeg_signed_principal_directions = np.zeros(
        signed_principal_directions.shape[:3], dtype=np.float32
    )
    for prin_dir_id in range(2):
        posdeg_signed_principal_directions[
            :, :, prin_dir_id
        ] = convert_angles_to_pos_degrees(
            signed_principal_directions[:, :, prin_dir_id]
        )

    features = [
        posdeg_signed_principal_directions,
        abs_eigvals,
        gradients_norms,
        posdeg_orientations,
    ]

    return features


######################
# Compute histograms #
######################


def rescale_value(values, kp_value, epsilon=1e-6):
    """
    Rescale vector values by keypoint value with broadcast.
    Vector value must be positive.
    If kp_value is null, divide by 1 instead.
    epsilon: float to avoid division by 0 (default is 1e-6)
    """
    if kp_value < epsilon:
        print("Warning: keypoint value is null, rescale by 1 instead")
        kp_value = 1  # could use another value such as 0.1 or 0.01 ? to better convey the fact that the value is null at that point
    return values / kp_value


def rotate_orientations(orientations, kp_orientation):
    """
    Rotate orientations with kp orientation in trigo order with broadcast.
    """
    rotated_orientations = (orientations - kp_orientation) % 360.0
    return rotated_orientations


def compute_vector_histogram(
    rescaled_values,
    rotated_orientations,
    kp_position,
    nb_bins=3,
    bin_radius=2,
    delta_angle=5.0,
    sigma=0,
):
    """
    Discretize neighborhood of keypoint into nb_bins bins, each of size 2*bin_radius+1, centered on keypoint.
    For each neighborhood, compute the histogram of the vectors orientations, weighted by vector values and distance to keypoint.
    Vector value must be positive.
    rescaled_values, rotated_orientations: numpy arrays of shape (height, width) (same as the image)
    kp_position: (x, y) int pixel position of keypoint in the image frame
    nb_bins: int odd number of bins of the neighborhood (default is 3)
    delta_angle: float size of the angular bins in degrees (default is 5)
    """
    # if sigma is null, define it with neighborhood_width
    neighborhood_width = nb_bins * (2 * bin_radius + 1)
    if sigma == 0:
        sigma = neighborhood_width / 2
    # compute the gaussian weight of the vectors
    g_weight = np.exp(-1 / (2 * sigma**2))

    # initialize the angular histograms
    nb_angular_bins = int(360 / delta_angle) + 1
    histograms = np.zeros((nb_bins, nb_bins, nb_angular_bins), dtype=np.float32)

    # loop over all pixels and add its contribution to the right angular histogram
    for bin_j in range(nb_bins):
        for bin_i in range(nb_bins):
            # compute the center of the neighborhood in the frame of the image
            x_center = kp_position[0] + (bin_j - nb_bins // 2) * (2 * bin_radius + 1)
            y_center = kp_position[1] + (bin_i - nb_bins // 2) * (2 * bin_radius + 1)
            # loop over all pixels of the neighborhood in the frame of the image
            for j in range(y_center - bin_radius, y_center + bin_radius + 1):
                for i in range(x_center - bin_radius, x_center + bin_radius + 1):
                    # compute the angle of the vector
                    angle = rotated_orientations[j, i]
                    # compute the gaussian weight of the vector
                    weight = g_weight * np.exp(
                        -((i - kp_position[0]) ** 2 + (j - kp_position[1]) ** 2)
                        / (2 * sigma**2)
                    )
                    # compute the bin of the angle
                    bin_angle = np.round(angle / delta_angle)
                    # add the weighted vector to the right histogram
                    histograms[bin_j, bin_i, bin_angle] += (
                        weight * rescaled_values[j, i]
                    )

            # normalize the histogram, now in percentage
            histograms[bin_j, bin_i, :] /= np.sum(histograms[bin_j, bin_i, :])
            histograms[bin_j, bin_i, :] *= 100.0

    return histograms


def rotate_point(x, y, angle, center_x, center_y):
    """
    Rotate a point counterclockwise by a given angle around a given origin, even if the vertical axis points downward.
    x, y: coordinates of the point
    angle: rotation angle in degrees
    center_x, center_y: coordinates of the rotation center
    return: new coordinates of the point
    """
    # Convert angle from degrees to radians
    angle = np.deg2rad(angle)

    # Translate point back to origin
    x -= center_x
    y -= center_y

    # Rotate point
    new_x = x * np.cos(angle) - y * np.sin(angle)
    new_y = x * np.sin(angle) + y * np.cos(angle)

    # Translate point back
    new_x += center_x
    new_y += center_y

    return new_x, new_y


def rotate_point_pixel(row, col, angle, center_row, center_col):
    """
    Rotate a point clockwise by a given angle around a given origin.
    row, col: coordinates of the point (in pixel coordinates)
    angle: rotation angle in degrees
    center_row, center_col: coordinates of the rotation center (in pixel coordinates)
    return: new coordinates of the point (in pixel coordinates)
    """
    # Convert angle from degrees to radians
    angle = np.deg2rad(angle)

    # Invert the row coordinate (because the y-axis points downward)
    row = -row
    center_row = -center_row

    # Translate point back to origin
    x = col - center_col
    y = row - center_row

    # Rotate point
    new_x = x * np.cos(angle) - y * np.sin(angle)
    new_y = x * np.sin(angle) + y * np.cos(angle)

    # Translate point back
    new_x += center_col
    new_y += center_row

    # Invert the new row coordinate back
    new_y = -new_y

    # round to int
    new_y = np.round(new_y).astype(np.int32)
    new_x = np.round(new_x).astype(np.int32)

    return new_y, new_x


def compute_rotated_vector_histogram(
    rescaled_values,
    rotated_orientations,
    kp_position,
    kp_gradient_orientation,
    nb_bins=3,
    bin_radius=2,
    delta_angle=5.0,
    sigma=0,
):
    """
    Discretize neighborhood of keypoint into nb_bins bins, each of size 2*bin_radius+1, centered on keypoint.
    Rotate neighborhood in the orientation of the keypoint
    For each spatial bin, compute the histogram of the vectors orientations, weighted by vector values and distance to keypoint.
    Vector value must be positive.
    rescaled_values, rotated_orientations: numpy arrays of shape (height, width) (same as the image)
    kp_position: (x, y) int pixel position of keypoint in the image frame
    nb_bins: int odd number of bins of the neighborhood (default is 3)
    delta_angle: float size of the angular bins in degrees (default is 5)
    """
    # if sigma is null, define it with neighborhood_width
    neighborhood_width = nb_bins * (2 * bin_radius + 1)
    if sigma == 0:
        sigma = neighborhood_width / 2
    # compute the gaussian weight of the vectors
    g_weight = np.exp(-1 / (2 * sigma**2))

    # initialize the angular histograms
    nb_angular_bins = int(360 / delta_angle) + 1
    histograms = np.zeros((nb_bins, nb_bins, nb_angular_bins), dtype=np.float32)

    # rotation is with angle +kp_gradient_orientation counterclockwise
    # loop over all pixels and add its contribution to the right angular histogram
    # recall i is the row index and j the column index
    for bin_i in range(nb_bins):
        for bin_j in range(nb_bins):
            # compute the center of the neighborhood in the frame of the image
            x_center = kp_position[0] + (bin_j - nb_bins // 2) * (2 * bin_radius + 1)
            y_center = kp_position[1] + (bin_i - nb_bins // 2) * (2 * bin_radius + 1)

            # loop over all pixels of the neighborhood
            # still in the frame of the image
            for i in range(y_center - bin_radius, y_center + bin_radius + 1):
                for j in range(x_center - bin_radius, x_center + bin_radius + 1):
                    # compute rotated pixel position in the image frame
                    # because rotate_point_pixel rotate a positive angle clockwise,
                    # we need to rotate with -kp_gradient_orientation
                    i_rot_img, j_rot_img = rotate_point_pixel(
                        i, j, -kp_gradient_orientation, kp_position[1], kp_position[0]
                    )

                    # compute the angle of the vector in the frame of the image
                    angle = rotated_orientations[i_rot_img, j_rot_img]
                    # compute the contribution of the vector in the frame of the image
                    contribution = rescaled_values[i_rot_img, j_rot_img]

                    # compute the gaussian weight of the vector with the distance in the image frame
                    weight = g_weight * np.exp(
                        -(
                            (i_rot_img - kp_position[1]) ** 2
                            + (j_rot_img - kp_position[0]) ** 2
                        )
                        / (2 * sigma**2)
                    )
                    # compute the bin of the angle
                    bin_angle = np.round(angle / delta_angle).astype(np.int32)

                    # add the weighted vector to the right histogram
                    histograms[bin_i, bin_j, bin_angle] += weight * contribution

            # normalize the histogram, now in percentage
            histograms[bin_i, bin_j, :] /= np.sum(histograms[bin_i, bin_j, :])
            histograms[bin_i, bin_j, :] *= 100.0

    return histograms


######################
# Compute descriptor #
######################


def compute_descriptor_histograms(
    overall_features, kp_position, nb_bins=3, bin_radius=2, delta_angle=5.0, sigma=0
):
    """
    Compute the histograms for the descriptor of a keypoint
    overall_features: list of features arrays of all pixels of the image
    kp_position: (x, y) int pixel position of keypoint in the image frame
    return descriptor_histograms: list of 3 histograms, each of shape (nb_bins, nb_bins, nb_angular_bins)
    """
    x_kp, y_kp = kp_position

    # unpack the features
    (
        posdeg_principal_directions,
        eigvals1pos,
        eigvals2pos,
        eigvals1neg,
        eigvals2neg,
        gradients_norms,
        posdeg_orientations,
    ) = overall_features

    # compute absolute value of keypoint eigenvalue for rescale
    kp_abs_eigval1 = eigvals1pos[y_kp, x_kp] + eigvals1neg[y_kp, x_kp]
    kp_abs_eigval2 = eigvals2pos[y_kp, x_kp] + eigvals2neg[y_kp, x_kp]

    # rescale vectors values by keypoint value
    rescaled_eigvals1pos = rescale_value(eigvals1pos, kp_abs_eigval1)
    rescaled_eigvals2pos = rescale_value(eigvals2pos, kp_abs_eigval2)
    rescaled_eigvals1neg = rescale_value(eigvals1neg, kp_abs_eigval1)
    rescaled_eigvals2neg = rescale_value(eigvals2neg, kp_abs_eigval2)

    # rotate vectors orientations with mean of kp orientation in trigo order
    # therefore it is a angle of reference for the keypoint that does not depend on the
    # order of the eigenvalues
    kp_prin_dirs = posdeg_principal_directions[y_kp, x_kp]
    mean_kp_prin_dir = 0.5 * (kp_prin_dirs[0] + kp_prin_dirs[1])
    rotated_prin_dirs = [
        rotate_orientations(
            posdeg_principal_directions[:, :, i],
            mean_kp_prin_dir,
        )
        for i in range(2)
    ]

    kp_orientation = posdeg_orientations[y_kp, x_kp]
    rotated_orientations = rotate_orientations(
        posdeg_orientations,
        kp_orientation,
    )

    # compute 1st positive eigenvalues histogram
    eig1pos_hist = compute_vector_histogram(
        rescaled_eigvals1pos,
        rotated_prin_dirs[0],
        kp_position,
        nb_bins,
        bin_radius,
        delta_angle,
        sigma,
    )

    # compute 2nd positive eigenvalues histogram
    eig2pos_hist = compute_vector_histogram(
        rescaled_eigvals2pos,
        rotated_prin_dirs[1],
        kp_position,
        nb_bins,
        bin_radius,
        delta_angle,
        sigma,
    )

    # compute 1st negative eigenvalues histogram
    eig1neg_hist = compute_vector_histogram(
        rescaled_eigvals1neg,
        rotated_prin_dirs[0],
        kp_position,
        nb_bins,
        bin_radius,
        delta_angle,
        sigma,
    )

    # compute 2nd negative eigenvalues histogram
    eig2neg_hist = compute_vector_histogram(
        rescaled_eigvals2neg,
        rotated_prin_dirs[1],
        kp_position,
        nb_bins,
        bin_radius,
        delta_angle,
        sigma,
    )

    # compute gradients histogram
    grad_hist = compute_vector_histogram(
        gradients_norms,
        rotated_orientations,
        kp_position,
        nb_bins,
        bin_radius,
        delta_angle,
        sigma,
    )

    # Add histograms of same sign eigenvalues
    eigpos_hist = eig1pos_hist + eig2pos_hist
    eigneg_hist = eig1neg_hist + eig2neg_hist

    # stack all histograms
    descriptor_histograms = [eigpos_hist, eigneg_hist, grad_hist]

    return descriptor_histograms


def compute_descriptor_histograms2(
    overall_features2, kp_position, nb_bins=3, bin_radius=2, delta_angle=5.0, sigma=0
):
    """
    Compute the histograms for the descriptor of a keypoint
    overall_features: list of features arrays of all pixels of the image
    kp_position: (x, y) int pixel position of keypoint in the image frame
    return descriptor_histograms: list of 3 histograms, each of shape (nb_bins, nb_bins, nb_angular_bins)
    1st histogram: first eigenvalues (highest signed value)
    2nd histogram: second eigenvalues
    3rd histogram: gradients
    """
    x_kp, y_kp = kp_position
    # unpack the features
    (
        posdeg_principal_directions,
        abs_eigvals,
        gradients_norms,
        posdeg_orientations,
    ) = overall_features2

    # compute absolute value of keypoint eigenvalue for rescale
    kp_abs_eigvals = abs_eigvals[y_kp, x_kp]

    # rescale vectors values by keypoint value
    rescaled_eigvals = np.zeros_like(abs_eigvals, dtype=np.float32)
    for eigval_id in range(2):
        rescaled_eigvals[:, :, eigval_id] = rescale_value(
            abs_eigvals[:, :, eigval_id], kp_abs_eigvals[eigval_id]
        )

    # rotate vectors orientations
    # principal directions
    kp_prin_dirs = posdeg_principal_directions[y_kp, x_kp]
    rotated_prin_dirs = np.zeros_like(posdeg_principal_directions, dtype=np.float32)
    for eigval_id in range(2):
        rotated_prin_dirs[:, :, eigval_id] = rotate_orientations(
            posdeg_principal_directions[:, :, eigval_id],
            kp_prin_dirs[eigval_id],
        )
    # gradients orientations
    kp_orientation = posdeg_orientations[y_kp, x_kp]
    rotated_orientations = rotate_orientations(
        posdeg_orientations,
        kp_orientation,
    )

    # compute the 2 eigenvalues histograms
    eigvals_hists = [
        compute_vector_histogram(
            rescaled_eigvals[:, :, eigval_id],
            rotated_prin_dirs[:, :, eigval_id],
            kp_position,
            nb_bins,
            bin_radius,
            delta_angle,
            sigma,
        )
        for eigval_id in range(2)
    ]

    # compute gradients histogram
    grad_hist = compute_vector_histogram(
        gradients_norms,
        rotated_orientations,
        kp_position,
        nb_bins,
        bin_radius,
        delta_angle,
        sigma,
    )

    # stack all histograms
    descriptor_histograms = [eigvals_hists[0], eigvals_hists[1], grad_hist]

    return descriptor_histograms


def compute_descriptor_histograms2_rotated(
    overall_features2, kp_position, nb_bins=3, bin_radius=2, delta_angle=5.0, sigma=0
):
    """
    Compute the histograms for the descriptor of a keypoint, with the rotated neighborhood.
    overall_features: list of features arrays of all pixels of the image
    kp_position: (x, y) int pixel position of keypoint in the image frame
    return descriptor_histograms: list of 3 histograms, each of shape (nb_bins, nb_bins, nb_angular_bins)
    1st histogram: first eigenvalues (highest signed value)
    2nd histogram: second eigenvalues
    3rd histogram: gradients
    """
    x_kp, y_kp = kp_position
    # unpack the features
    (
        posdeg_principal_directions,
        abs_eigvals,
        gradients_norms,
        posdeg_orientations,
    ) = overall_features2

    # compute absolute value of keypoint eigenvalue for rescale
    kp_abs_eigvals = abs_eigvals[y_kp, x_kp]

    # rescale vectors values by keypoint value
    rescaled_eigvals = np.zeros_like(abs_eigvals, dtype=np.float32)
    for eigval_id in range(2):
        rescaled_eigvals[:, :, eigval_id] = rescale_value(
            abs_eigvals[:, :, eigval_id], kp_abs_eigvals[eigval_id]
        )

    # rotate vectors orientations
    # principal directions
    kp_prin_dirs = posdeg_principal_directions[y_kp, x_kp]
    rotated_prin_dirs = np.zeros_like(posdeg_principal_directions, dtype=np.float32)
    for eigval_id in range(2):
        rotated_prin_dirs[:, :, eigval_id] = rotate_orientations(
            posdeg_principal_directions[:, :, eigval_id],
            kp_prin_dirs[eigval_id],
        )
    # gradients orientations
    kp_orientation = posdeg_orientations[y_kp, x_kp]
    rotated_orientations = rotate_orientations(
        posdeg_orientations,
        kp_orientation,
    )

    # compute the 2 eigenvalues histograms
    eigvals_hists = [
        compute_rotated_vector_histogram(
            rescaled_eigvals[:, :, eigval_id],
            rotated_prin_dirs[:, :, eigval_id],
            kp_position,
            kp_orientation,
            nb_bins,
            bin_radius,
            delta_angle,
            sigma,
        )
        for eigval_id in range(2)
    ]

    # compute gradients histogram
    grad_hist = compute_rotated_vector_histogram(
        gradients_norms,
        rotated_orientations,
        kp_position,
        kp_orientation,
        nb_bins,
        bin_radius,
        delta_angle,
        sigma,
    )

    # stack all histograms
    descriptor_histograms = [eigvals_hists[0], eigvals_hists[1], grad_hist]

    return descriptor_histograms


############################
# Descriptor Visualization #
############################


def display_histogram(histogram):
    """
    Display a histogram. On the abscissa is the angle in degrees, on the ordinate is the value.
    histogram: numpy array of shape (nb_angular_bins,)
    """
    nb_angular_bins = histogram.shape[0]
    angles = np.linspace(0.0, 360.0, nb_angular_bins)
    plt.hist(angles, weights=histogram, bins=nb_angular_bins)
    plt.show()


def display_spatial_histograms(histograms, title="Spatial Histograms"):
    """
    Display all spatial histograms around a keypoint.
    histograms: array of shape (nb_bins, nb_bins, nb_angular_bins)
    """
    nb_bins = histograms.shape[0]
    nb_angular_bins = histograms.shape[2]
    angles = np.linspace(0.0, 360.0, nb_angular_bins)

    # make a figure with nb_bins * nb_bins subplots
    fig, axs = plt.subplots(nb_bins, nb_bins, figsize=(nb_bins * 8, nb_bins * 8))

    # loop over all subplots
    for bin_j in range(nb_bins):
        for bin_i in range(nb_bins):
            # display the histogram
            axs[bin_j, bin_i].bar(
                angles,
                histograms[bin_j, bin_i, :],
            )
            # set the title of the subplot
            axs[bin_j, bin_i].set_title(
                f"Bin ({bin_j}, {bin_i})",
            )
            # add axis labels
            axs[bin_j, bin_i].set_xlabel("Angle (degrees)")

    # set the title of the figure
    fig.suptitle(title)

    # Adjust the space between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    # plt.show()


def display_descriptor(
    descriptor_histograms,
    descriptor_name="Descriptor",
    values_names=["positive eigenvalues", "negative eigenvalues", "gradients"],
):
    """
    Display the descriptor of a keypoint.
    descriptor_histograms: list of 3 histograms, each of shape (nb_bins, nb_bins, nb_angular_bins)
    """

    for id_value in range(len(values_names)):
        display_spatial_histograms(
            descriptor_histograms[id_value],
            title=f"{descriptor_name}, {values_names[id_value]}",
        )

    plt.show()


#######################
# Descriptor Matching #
#######################


def flatten_descriptor(descriptor_histograms):
    """
    Flatten the descriptor histograms into a 1D vector.
    descriptor_histograms: list of 3 histograms, each of shape (nb_bins, nb_bins, nb_angular_bins)
    return descriptor: 1D numpy array
    """
    descriptor = np.concatenate(
        [
            descriptor_histograms[0].flatten(),
            descriptor_histograms[1].flatten(),
            descriptor_histograms[2].flatten(),
        ]
    )
    return descriptor


def compute_descriptor_distance(descriptor1, descriptor2):
    """
    Compute the euclidean distance between 2 descriptors.
    descriptor1, descriptor2: 1D numpy arrays
    return distance: float
    """
    distance = np.linalg.norm(descriptor1 - descriptor2)
    return distance
