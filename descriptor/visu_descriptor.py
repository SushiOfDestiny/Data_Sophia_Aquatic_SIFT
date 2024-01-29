import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, fftconvolve
import scipy.ndimage as ndimage

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
                # display the histogram
                col_id = id_hist * nb_bins + bin_i
                axs[bin_j, col_id].bar(
                    angles,
                    hists[id_hist][bin_j, bin_i, :],
                    color=colors[id_hist],
                )
                # set the title of the subplot
                axs[bin_j, col_id].set_title(
                    f"Bin ({bin_j}, {bin_i})",
                )
                # add axis labels
                axs[bin_j, col_id].set_xlabel("Angle (degrees)")

    # set the title of the figure
    fig.suptitle(hist_title)

    # Create a legend for the colors
    plt.legend(["Histogram 1", "Histogram 2"], title="Histograms", loc="upper right")

    # Adjust the space between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    # plt.show()


def display_matched_descriptors(
    desc1,
    desc2,
    title="Descriptors of matched keypoints",
    hist_titles=["1st eigenvalues", "2nd eigenvalues", "gradients"],
):
    """
    Display 3 plots each showing side by side corresponding histograms in the 2 descriptors.
    desc1: list of 3 histograms, each of shape (nb_bins, nb_bins, nb_angular_bins)
    desc2: same
    """
    for hist_idx in range(3):
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        axs[0].imshow(desc1[hist_idx])
        axs[0].set_title(f"Descriptor 1, {hist_idx}")
        axs[1].imshow(desc2[hist_idx])
        axs[1].set_title(f"Descriptor 2, {hist_idx}")
        fig.suptitle(title)

    plt.show()
