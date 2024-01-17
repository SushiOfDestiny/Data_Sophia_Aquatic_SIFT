import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, fftconvolve

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
    position: (y, x) int pixel position of point
    epsilon: float to avoid division by 0 (default is 1e-6)
    return orientation: float32 orientation of the keypoint in degrees, within [0, 360[
    """
    y, x = position
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
    Convert angles in radians to positive degrees in [0, 360[
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
            orientations[y, x] = compute_orientation(g_img, (y, x))
    return orientations


#######################
# Gaussians averaging #
#######################


def create_1D_gaussian_kernel(sigma, size=0):
    """
    Create a 1D Gaussian kernel of a given sigma
    sigma: float
    Return: 1D numpy array
    """
    if size == 0:
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


def compute_angle(v1, v2):
    """
    Compute the angle between 2 vectors, in radiants
    """
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return angle


def compute_principal_directions(eigvects):
    """
    compute principal directions as the angle between eigenvectors and horizontal axis of the image
    eigvects: numpy array of shape (height, width, 2, 2) of float32
    return principal_directions: numpy array of shape (height, width, 2) of float32 with angles in radians
    """
    horiz_vect = np.array([0, 1], dtype=np.float32)
    principal_directions = np.zeros(eigvects.shape[:3], dtype=np.float32)
    for id_dir in range(2):
        principal_directions[:, :, id_dir] = compute_angle(
            eigvects[:, :, id_dir], horiz_vect
        )
    return principal_directions


def split_eigenvalues(eigvals):
    # separate for each index of eigenvalue, the eigenvalue according to its sign
    eigvals1pos = eigvals[:, :, 0] * (eigvals[:, :, 0] > 0)
    eigvals1neg = eigvals[:, :, 0] * (eigvals[:, :, 0] < 0) * (-1)
    eigvals2pos = eigvals[:, :, 1] * (eigvals[:, :, 1] > 0)
    eigvals2neg = eigvals[:, :, 1] * (eigvals[:, :, 1] < 0) * (-1)

    # return list of features in specific order
    return [eigvals1pos, eigvals2pos, eigvals1neg, eigvals2neg]


def compute_features_overall(g_img, border_size=1):
    """
    Compute useful features for all pixels of a grayscale image.
    Return a list of pairs of features arrays, each pair containing the feature values and the feature orientations.
    All angular features are in degrees in [0, 360[.
    """
    # compute eigenvalues and eigenvectors of the Hessian matrix
    eigvals, eigvects, gradients = vh.compute_hessian_gradient_subimage(
        g_img, border_size
    )
    # compute gradients norms
    gradients_norms = np.linalg.norm(gradients, axis=2)

    # compute principal directions of the Hessian matrix
    principal_directions = compute_principal_directions(eigvects)

    # compute orientations of the gradients in degrees
    orientations = compute_orientations(g_img, border_size)

    # convert and rescale angles in [0, 360[
    posdeg_orientations = convert_angles_to_pos_degrees(orientations)
    posdeg_principal_directions = convert_angles_to_pos_degrees(principal_directions)

    # separate eigenvalues according to their sign
    splitted_eigvals = split_eigenvalues(eigvals)
    eigvals1pos, eigvals2pos, eigvals1neg, eigvals2neg = splitted_eigvals

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


######################
# Compute histograms #
######################


def rescale_value(values, kp_value):
    """
    Rescale vector values by keypoint value.
    Vector value must be positive.
    """
    return values / kp_value


def rotate_orientations(orientations, kp_position):
    """
    Rotate orientations with kp orientation in trigo order
    """
    kp_orientation = orientations[kp_position]
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
    nb_bins: int odd number of bins of the neighborhood (default is 3)
    delta_angle: float size of the angular bins in degrees (default is 5)
    """
    # if sigma is null, define it with neighborhood_size
    neighborhood_size = nb_bins * (2 * bin_radius + 1)
    if sigma == 0:
        sigma = neighborhood_size / 2
    # compute the gaussian weight of the vectors
    g_weight = np.exp(-1 / (2 * sigma**2))

    # initialize the angular histograms
    nb_angular_bins = int(360 / delta_angle) + 1
    histograms = np.zeros((nb_bins, nb_bins, nb_angular_bins), dtype=np.float32)

    # loop over all pixels and add its contribution to the right angular histogram
    for bin_j in range(nb_bins):
        for bin_i in range(nb_bins):
            # compute the center of the neighborhood
            y = kp_position[0] + (bin_j - nb_bins // 2) * (2 * bin_radius + 1)
            x = kp_position[1] + (bin_i - nb_bins // 2) * (2 * bin_radius + 1)
            # loop over all pixels of the neighborhood
            for j in range(y - bin_radius, y + bin_radius + 1):
                for i in range(x - bin_radius, x + bin_radius + 1):
                    # compute the angle of the vector
                    angle = rotated_orientations[j, i]
                    # compute the gaussian weight of the vector
                    weight = g_weight * np.exp(
                        -((j - kp_position[0]) ** 2 + (i - kp_position[1]) ** 2)
                        / (2 * sigma**2)
                    )
                    # compute the bin of the angle
                    bin_angle = int(angle / delta_angle)
                    # add the weighted vector to the right histogram
                    histograms[bin_j, bin_i, bin_angle] += (
                        weight * rescaled_values[j, i]
                    )
    return histograms


######################
# Compute descriptor #
######################


def compute_descriptor_histograms(overall_features, kp_position):
    """
    Compute the histograms for the descriptor of a keypoint
    overall_features: list of features arrays of all pixels of the image
    kp_position: (y, x) int pixel position of keypoint
    return descriptor_histograms: list of 3 histograms, each of shape (nb_bins, nb_bins, nb_angular_bins)
    """
    y_kp, x_kp = kp_position
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

    # rotate vectors orientations with kp orientation in trigo order
    rotated_prin_dirs = [
        rotate_orientations(
            posdeg_principal_directions[:, :, i],
            kp_position,
        )
        for i in range(2)
    ]

    rotated_orientations = rotate_orientations(
        posdeg_orientations,
        kp_position,
    )

    # compute 1st positive eigenvalues histogram
    eig1pos_hist = compute_vector_histogram(
        rescaled_eigvals1pos,
        rotated_prin_dirs[0],
        kp_position,
    )

    # compute 2nd positive eigenvalues histogram
    eig2pos_hist = compute_vector_histogram(
        rescaled_eigvals2pos,
        rotated_prin_dirs[1],
        kp_position,
    )

    # compute 1st negative eigenvalues histogram
    eig1neg_hist = compute_vector_histogram(
        rescaled_eigvals1neg,
        rotated_prin_dirs[0],
        kp_position,
    )

    # compute 2nd negative eigenvalues histogram
    eig2neg_hist = compute_vector_histogram(
        rescaled_eigvals2neg,
        rotated_prin_dirs[1],
        kp_position,
    )

    # compute gradients histogram
    grad_hist = compute_vector_histogram(
        gradients_norms,
        rotated_orientations,
        kp_position,
    )

    # Add histograms of same sign eigenvalues
    eigpos_hist = eig1pos_hist + eig2pos_hist
    eigneg_hist = eig1neg_hist + eig2neg_hist

    # stack all histograms
    descriptor_histograms = [eigpos_hist, eigneg_hist, grad_hist]

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
    x = np.linspace(0.0, 360.0, nb_angular_bins)
    plt.hist(x, weights=histogram, bins=nb_angular_bins)
    plt.show()


def display_spatial_histograms(histograms, title="Spatial Histograms"):
    """
    Display all spatial histograms around a keypoint.
    histograms: array of shape (nb_bins, nb_bins, nb_angular_bins)
    """
    nb_bins = histograms.shape[0]
    nb_angular_bins = histograms.shape[2]
    x = np.linspace(0.0, 360.0, nb_angular_bins)

    # make a figure with nb_bins * nb_bins subplots
    fig, axs = plt.subplots(nb_bins, nb_bins, figsize=(nb_bins * 8, nb_bins * 8))

    # loop over all subplots
    for bin_j in range(nb_bins):
        for bin_i in range(nb_bins):
            # display the histogram
            axs[bin_j, bin_i].bar(
                x,
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

    plt.show()


def display_descriptor(descriptor_histograms):
    """
    Display the descriptor of a keypoint.
    descriptor_histograms: list of 3 histograms, each of shape (nb_bins, nb_bins, nb_angular_bins)
    """
    values_names = ["positive eigenvalues", "negative eigenvalues", "gradients"]
    for id_value in range(len(values_names)):
        display_spatial_histograms(
            descriptor_histograms[id_value],
            title=values_names[id_value],
        )
