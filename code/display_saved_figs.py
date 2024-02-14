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

fig_name = "rocks_2_10_deg_y_525_525_225_225_x_675_600_300_300_nbins_3_brad_2_nangbins_73_sig0_min_Random unfiltered by blender  matches_523_multi"
fig_path = f"filtered_keypoints/{fig_name}.pkl"

if __name__ == "__main__":

    with open(fig_path, "rb") as f:
        fig = pickle.load(f)

    plt.show(block=True)
