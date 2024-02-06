import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, fftconvolve
import scipy.ndimage as ndimage

import visu_hessian as vh
import descriptor as desc

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
            # get the current subplot
            ax = axs if nb_bins == 1 else axs[bin_j, bin_i]

            # display the histogram
            ax.bar(angles, histograms[bin_j, bin_i, :])

            # set the title of the subplot
            ax.set_title(f"Bin ({bin_j}, {bin_i})")

            # add axis labels
            ax.set_xlabel("Angle (degrees)")

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


def display_matched_histograms(
    hist1, hist2, hist_title="first eigenvalues", figsize_ppt=2, figsize_width=8
):
    """
    Display 2 histograms side by side.
    hist1: numpy array of shape (nb_bins, nb_bins, nb_angular_bins)
    hist2: same
    """
    hists = [hist1, hist2]
    nb_bins = hist1.shape[0]
    nb_angular_bins = hist1.shape[2]
    angles = np.linspace(0.0, 360.0, nb_angular_bins)
    fig, axs = plt.subplots(
        nb_bins,
        2 * nb_bins,
        figsize=(figsize_ppt * nb_bins * figsize_width, nb_bins * figsize_width),
    )

    # Define colors for each histogram
    colors = ["blue", "orange"]

    # loop over all subplots
    for id_hist in range(2):
        for bin_j in range(nb_bins):
            for bin_i in range(nb_bins):
                # calculate column id
                col_id = id_hist * nb_bins + bin_i

                # get the current subplot
                ax = axs[bin_j, col_id] if nb_bins > 1 else axs[col_id]

                # display the histogram
                ax.bar(
                    angles,
                    hists[id_hist][bin_j, bin_i, :],
                    color=colors[id_hist],
                )
                # set the title of the subplot
                ax.set_title(
                    f"Bin ({bin_j}, {bin_i})",
                )
                # add axis labels
                ax.set_xlabel("Angle (degrees)")

    # set the title of the figure
    fig.suptitle(hist_title)

    # Create a legend for the colors
    plt.legend(["Histogram 1", "Histogram 2"], title="Histograms", loc="upper right")

    # Adjust the space between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    # plt.show()


###########################
# Visualisation Pipelines #
###########################


def descriptor_visualisation_pipeline(
    kps,
    float_ims,
    nb_bins=3,
    bin_radius=2,
    border_size=1,
    delta_angle=5.0,
    sigma=0,
    normalization_mode="global",
):
    """
    Compute and display the descriptor of keypoints side by side
    """
    # get keypoints positions
    kps_pos = [tuple(np.round(kp.pt).astype(int)) for kp in kps]

    # compute neighborhood radius
    neigh_radius = (2 * bin_radius + 1) * nb_bins // 2
    larger_neigh_radius = neigh_radius + 3 * border_size

    # crop the subimages around keypoints
    crop_ims = [
        vh.crop_image_around_keypoint(float_ims[i], kps_pos[i], larger_neigh_radius)
        for i in range(2)
    ]

    # compute zoomed positions
    zoomed_pos = [larger_neigh_radius, larger_neigh_radius]

    # compute overall features of the neighborhoods
    zoomed_features = [
        desc.compute_features_overall_abs(crop_ims[i], border_size) for i in range(2)
    ]

    descriptor_histograms_kps = [
        desc.compute_descriptor_histograms_1_2_rotated(
            overall_features_1_2=zoomed_features[i],
            kp_position=zoomed_pos,
            nb_bins=nb_bins,
            bin_radius=bin_radius,
            delta_angle=delta_angle,
            sigma=sigma,
            normalization_mode=normalization_mode,
        )
        for i in range(2)
    ]

    values_names = [
        "first principal direction",
        "second principal direction",
        "gradient directions",
    ]

    for id_feature in range(3):
        display_matched_histograms(
            hist1=descriptor_histograms_kps[0][id_feature],
            hist2=descriptor_histograms_kps[1][id_feature],
            hist_title=values_names[id_feature],
        )

    plt.show()
