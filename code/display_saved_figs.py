# Testing pipeline after Blender script execution
import os
import sys

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import descriptor as desc
import visu_hessian as vh
import visu_descriptor as visu_desc

from computation_pipeline_hyper_params import *

from filenames_creation import *

import compute_desc_img as cp_desc

import pickle


def display_pickled_figs(fig_paths_list, show_figs=True):
    """
    Display the figures pickled in the paths in fig_paths_list.
    """
    for fig_path in fig_paths_list:
        with open(fig_path, "rb") as f:
            fig = pickle.load(f)
            fig.canvas.draw()
    if show_figs:
        plt.show(block=True)


if __name__ == "__main__":
    fig_names = [
        "rocks_2_10_deg_y_525_525_225_225_x_675_600_300_300_nbins_3_brad_2_nangbins_73_sig0_min_Random unfiltered by blender  matches_523_multi",
        "rocks_2_10_deg_y_525_525_225_225_x_675_600_300_300_nbins_3_brad_2_nangbins_73_sig0_sift_min_Random unfiltered by blender sift matches_678_multi",
    ]
    fig_paths = [f"filtered_keypoints/{fig_name}.pkl" for fig_name in fig_names]

    display_pickled_figs(fig_paths, show_figs=True)
