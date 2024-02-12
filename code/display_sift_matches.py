import os
import sys
from datetime import datetime

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

from descriptor import (
    compute_descriptor_histograms_1_2_rotated,
    flatten_descriptor,
    compute_features_overall_abs,
)


if __name__ == "__main__":

    # load images
    ims = [
        cv.imread(
            f"{relative_path}/{img_folder}/{im_names[id_image]}.{im_ext}",
            cv.IMREAD_GRAYSCALE,
        )
        for id_image in range(2)
    ]
    float_ims = np.load(f"{blurred_imgs_filename}.npy")

    # load all computed objects
    # load unfiltered keypoints coordinates
    kps_coords = [
        np.load(f"{descrip_path}/{kp_coords_filenames[id_image]}.npy")
        for id_image in range(2)
    ]

    # load unfiltered keypoints, matches and index of good matches
    kps = [
        load_keypoints(f"{matches_path}/{kp_filenames[id_image]}")
        for id_image in range(2)
    ]
    matches = load_matches(f"{matches_path}/{matches_filename}")

    # Load matches filtered by blender
    correct_matches_idxs = np.load(
        f"{matches_path}/{correct_matches_idxs_filename}.npy"
    )

    # filter good matches according to blender
    good_matches, bad_matches = dm.compute_good_and_bad_matches(
        matches=matches, good_matches_kps_idx=correct_matches_idxs
    )

    # print general info about proportions of keypoints and matches

    dm.print_general_kp_matches_infos(float_ims, kps, matches, good_matches, epsilon)

    # print stats about good matches and bad matches
    # look at the distances of the good and bad matches
    print("Statistics about the distances of the good sift matches")
    dm.print_distance_infos(good_matches)

    print("Statistics about the distances of the bad sift matches")
    dm.print_distance_infos(bad_matches)

    # # Compute descriptor on good and bad SIFT keypoints on image 0
    # # Requires computing preprocessed images and features first

    # before = datetime.now()
    # print(f"feat computation beginning for image {0}:", before)
    # overall_features = compute_features_overall_abs(
    #     float_ims[0], border_size=border_size
    # )
    # after = datetime.now()
    # print(f"feat computation end for image {0}", after)
    # print(f"feat compute time for image {0}", after - before)
    # print(
    #     f"shape of overall_features[0] of image {0}",
    #     overall_features[0].shape,
    # )

    # # Compute good and bad descriptors

    # descs_good_matches = np.zeros(shape=(len(good_matches), 3*nb_bins*nb_bins*nb_angular_bins))
    # for i, match in enumerate(good_matches):
    #     descs_good_matches[i] = flatten_descriptor(
    #         compute_descriptor_histograms_1_2_rotated(
    #             overall_features_1_2 = overall_features,
    #             kp_position = kps_coords[0][match[0].queryIdx],
    #             nb_bins=nb_bins,
    #             bin_radius=bin_radius,
    #             delta_angle=delta_angle,
    #             sigma=0,
    #             normalization_mode="global",
    #         )
    #     )
    # avg_good_desc = np.mean(descs_good_matches, axis=0)
    # visu_desc.display_descriptor(unflatten_descriptor(avg_good_desc, nb_bins, nb_angular_bins), show_plot=False)

    # descs_bad_matches = np.zeros(shape=(len(bad_matches), 3*nb_bins*nb_bins*nb_angular_bins))
    # for i, match in enumerate(bad_matches):
    #     descs_bad_matches[i] = flatten_descriptor(
    #         compute_descriptor_histograms_1_2_rotated(
    #             overall_features_1_2 = overall_features,
    #             kp_position = kps_coords[0][match[0].queryIdx],
    #             nb_bins=nb_bins,
    #             bin_radius=bin_radius,
    #             delta_angle=delta_angle,
    #             sigma=0,
    #             normalization_mode="global",
    #         )
    #     )
    # avg_bad_desc = np.mean(descs_bad_matches, axis=0)
    # visu_desc.display_descriptor(unflatten_descriptor(avg_bad_desc, nb_bins, nb_angular_bins), show_plot=True)

    # look at some matches

    # chosen_matches_idx = [0, 1]
    # for match_idx in chosen_matches_idx:
    #     # display 1 match, object here is not DMatch, but a couple of DMatch, as Sift returns
    #     # we get here only the Dmatch
    #     chosen_Dmatch = good_matches[match_idx][0]

    #     # display the match
    #     dm.display_match(
    #         ims,
    #         chosen_Dmatch,
    #         kps_coords,
    #         show_plot=False,
    #         # save_path=filtered_kp_path,  # comment or pass None to not save the image
    #         filename_prefix=f"{correct_match_filename_prefix}",
    #         dpi=800,
    #         im_names=im_names,
    #     )

    # # display topological properties
    # chosen_kps = [kps[0][chosen_Dmatch.queryIdx], kps[1][chosen_Dmatch.trainIdx]]
    # vh.topological_visualization_pipeline(
    #     kps=chosen_kps,
    #     uint_ims=ims,
    #     float_ims=float_ims,
    #     zoom_radius=20,
    #     show_directions=False,
    #     show_gradients=False,
    #     show_plot=False,
    # )

    # plt.show()
