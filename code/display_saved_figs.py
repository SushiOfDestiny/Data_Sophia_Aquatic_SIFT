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
    # find all pickle filenames in the directory containing "higher"
    fig_names = [
        f[:-4]
        for f in os.listdir("filtered_keypoints")
        if f.endswith(".pkl") and ("higher" in f or "irl_rock1" in f)
    ]

    fig_names = [
        "irl_rock1_y_525_745_800_800_x_880_1110_1500_1500_nbins_3_brad_2_nangbins_73_sig0_filt_70_min_Top 3000 minimal unfiltered by blender  matches_588_multi",
        # "rocks_2_10_deg_higher_distance_y_450_450_300_300_x_590_590_580_580_nbins_3_brad_2_nangbins_73_sig0_filt_60_min_Random minimal  matches, top 2% of pixels in subimage 1_99_multi",
        # "rocks_2_10_deg_higher_distance_y_450_450_300_300_x_590_590_580_580_nbins_3_brad_2_nangbins_73_sig0_sift_min_250 Random minimal sift matches, among the top 3% of pixels in subimage 1_215_multi",
        # "rocks_2_7_deg_higher_distance_y_450_450_300_300_x_650_560_580_580_nbins_3_brad_2_nangbins_73_sig0_filt_70_min_Random good  matches_192_multi",
        # "rocks_2_7_deg_higher_distance_y_450_450_300_300_x_650_560_580_580_nbins_3_brad_2_nangbins_73_sig0_filt_70_min_prefiltered_pixels",
        # "rocks_2_7_deg_higher_distance_y_450_450_300_300_x_650_560_580_580_nbins_3_brad_2_nangbins_73_sig0_filt_70_min_prefiltered_pixels_mask",
        # "rocks_2_7_deg_higher_distance_y_450_450_300_300_x_650_560_580_580_nbins_3_brad_2_nangbins_73_sig0_filt_70_min_250 Random minimal  matches, among the top 2% of pixels in subimage 1_338_multi",
        # "rocks_2_7_deg_higher_distance_y_450_450_300_300_x_650_560_580_580_nbins_3_brad_2_nangbins_73_sig0_sift_min_250 Random minimal sift matches, among the top 2% of pixels in subimage 1_394_multi",
        # "rocks_2_7_deg_higher_distance_y_450_450_300_300_x_650_560_580_580_nbins_3_brad_2_nangbins_73_sig0_sift_min_250 Random minimal sift matches, among the top 2% of pixels in subimage 1_637_multi",
        # "rocks_2_10_deg_higher_distance_y_450_450_300_300_x_590_590_580_580_nbins_3_brad_2_nangbins_73_sig0_filt_60_min_Random minimal  matches, top 2% of pixels in subimage 1_99_multi",
        # "rocks_2_10_deg_higher_distance_y_450_450_300_300_x_590_590_580_580_nbins_3_brad_2_nangbins_73_sig0_sift_min_250 Random minimal sift matches, among the top 3% of pixels in subimage 1_215_multi",
        # "rocks_2_10_deg_higher_distance_y_450_450_300_300_x_590_590_580_580_nbins_3_brad_2_nangbins_73_sig0_sift_min_250 Random minimal sift matches, among the top 2% of pixels in subimage 1_125_multi",
        # "rocks_2_10_deg_higher_distance_y_450_450_300_300_x_590_590_580_580_nbins_3_brad_2_nangbins_73_sig0_filt_60_min_Random good  matches_755_multi",
    ]

    fig_paths = [f"filtered_keypoints/{fig_name}.pkl" for fig_name in fig_names]

    display_pickled_figs(fig_paths, show_figs=True)
