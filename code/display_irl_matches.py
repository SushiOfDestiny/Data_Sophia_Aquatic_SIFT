# Goal of this script is to visualise some matches on real images
import os
import sys

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from descriptor import unflatten_descriptor
import visu_hessian as vh
import visu_descriptor as visu_desc

sys.path.append("../matching")
from saving import load_matches, load_keypoints

from computation_pipeline_hyper_params import *

from filenames_creation import *

import display_matches as dm
import compute_desc_img as cp_desc
import descriptor as desc
from matplotlib.ticker import PercentFormatter


if __name__ == "__main__":

    plot_hist = False

    # load images
    ims = [
        cv.imread(
            f"{relative_path}/{img_folder}/{im_names[id_image]}.{im_ext}",
            cv.IMREAD_GRAYSCALE,
        )
        for id_image in range(2)
    ]
    float_ims = np.load(f"{blurred_imgs_filename}.npy")

    # load cropped float images
    cropped_float_ims = [
        np.load(
            f"{original_imgs_path_prefix}/{cropped_float_ims_filenames[id_image]}.npy"
        )
        for id_image in range(2)
    ]


    # load all computed objects

    # load unfiltered keypoints coordinates
    kps_coords = [
        np.load(f"{descrip_path}/{kp_coords_filenames[id_image]}.npy")
        for id_image in range(2)
    ]

    # load unfiltered keypoints, matches and index of good matches
    kps = [
        load_keypoints(f"{matches_path}/{kp_filenames[id_image]}.txt")
        for id_image in range(2)
    ]
    matches = load_matches(f"{matches_path}/{matches_filename}.txt")



    matches_kps = [
        [kps_coords[0][match[0].queryIdx] for match in matches],
        [kps_coords[1][match[0].trainIdx] for match in matches],
    ]


    matches_kps_idx = [
        np.array(
            [
                (kp[1] - y_starts[id_image]) * x_lengths[id_image]
                + (kp[0] - x_starts[id_image])
                for kp in matches_kps[id_image]
            ]
        )
        for id_image in range(2)
    ]


    # look at statistics about the distances of the matches
    print("Statistics about the distances of the matches")
    dm.print_distance_infos(matches)

    # display random matches
    max_nb_matches_to_display = 250
    nb_rd_unfiltered_matches_to_display = min(len(matches), max_nb_matches_to_display)
    rd_idx_unfiltered = np.random.choice(
        len(matches), nb_rd_unfiltered_matches_to_display
    )
    rd_matches_unfiltered = [matches[i] for i in rd_idx_unfiltered]
    dm.display_matches(
        ims,
        rd_matches_unfiltered,
        kps_coords,
        show_plot=False,
        im_names=im_names,
        plot_title_prefix=f"Random unfiltered by blender {sift_radical[1:] if use_sift else ''} matches",
    )

    plt.show()


    if not use_sift:
        # Load descriptors
        descs = [
            np.load(
                f"{descrip_path}/{descrip_filenames[id_image]}.npy",
            )
            for id_image in range(2)
        ]

        descs_ims = [descs[id_image][matches_kps_idx[id_image]] for id_image in range(2)]
        
        # Look at the averaged descriptor
        avg_descs = [np.mean(descs_ims[id_image], axis=0) for id_image in range(2)]

        descs_names = [
            f"Averaged descriptor of matches for {im_names[id_image]}\n with nb_bins={nb_bins}, bin_radius={bin_radius}, delta_angle={delta_angle} and sigma={sigma}"
            for id_image in range(2)
        ]

        if plot_hist:
            overall_features_ims = [
                desc.compute_features_overall_abs(
                    float_ims[id_image], border_size=border_size
                )
            for id_image in range(2)
            ]
            for id_image in range(2):
                mean_abs_curvs = cp_desc.compute_mean_abs_curv_arr(overall_features_ims[id_image][1])
                plt.figure()
                hist = plt.hist(x=mean_abs_curvs, nbins=100, weights=np.ones(mean_abs_curvs)/len(mean_abs_curvs))
                plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
                plt.xlabel("Moyenne des valeurs absolues des courbures")
                plt.ylabel("Pourcentage")


        # for id_image in range(2):
        #     visu_desc.display_descriptor(
        #         descriptor_histograms=unflatten_descriptor(avg_descs[id_image]),
        #         descriptor_name=descs_names[id_image],
        #         show_plot=False,
        #     )

        plt.show()

    
