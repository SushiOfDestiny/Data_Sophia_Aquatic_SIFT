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

if __name__ == "__main__":
    # load images
    ims = [
        cv.imread(
            f"{relative_path}/{img_folder}/{im_names[id_image]}.{im_ext}",
            cv.IMREAD_GRAYSCALE,
        )
        for id_image in range(2)
    ]
    float_ims = np.load(f"{blurred_imgs_path}.npy")

    # load all computed objects

    # load unfiltered keypoints coordinates
    kps_coords = [
        np.load(f"{descrip_path}/{kp_coords_filenames[id_image]}")
        for id_image in range(2)
    ]
    descs = [
        np.load(
            f"{descrip_path}/{descrip_filenames[id_image]}",
        )
        for id_image in range(2)
    ]

    # load unfiltered keypoints, matches and index of good matches
    kps = [
        load_keypoints(f"{matches_path}/{kp_filenames[id_image]}")
        for id_image in range(2)
    ]
    matches = load_matches(f"{matches_path}/{matches_filename}")

    # sort matches by distance
    sorted_matches = sorted(matches, key=lambda x: x[0].distance)

    # look at some matches
    chosen_matches_idx = list(range(10))
    for match_idx in chosen_matches_idx:
        # display 1 match, object here is not DMatch, but a couple of DMatch, as Sift returns
        # we get here only the Dmatch
        chosen_Dmatch = sorted_matches[match_idx][0]

        # display the match
        dm.display_match(
            ims,
            chosen_Dmatch,
            kps_coords,
            show_plot=False,
            # save_path=filtered_kp_path,  # comment or pass None to not save the image
            filename_prefix=correct_match_filename_prefix,
            dpi=800,
            im_names=im_names,
        )

        # display topological properties
        chosen_kps = [kps[0][chosen_Dmatch.queryIdx], kps[1][chosen_Dmatch.trainIdx]]
        vh.topological_visualization_pipeline(
            kps=chosen_kps,
            uint_ims=ims,
            float_ims=float_ims,
            zoom_radius=20,
            show_directions=False,
            show_gradients=False,
            show_plot=False,
        )

    plt.show()

    # Look at the averaged descriptor
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

    descs_ims = [descs[id_image][matches_kps_idx[id_image]] for id_image in range(2)]
    avg_descs = [np.mean(descs_ims[id_image], axis=0) for id_image in range(2)]

    descs_names = [
        f"Averaged descriptor of matches for {im_names[id_image]}\n with nb_bins={nb_bins}, bin_radius={bin_radius}, delta_angle={delta_angle} and sigma={sigma}"
        for id_image in range(2)
    ]

    for id_image in range(2):
        visu_desc.display_descriptor(
            descriptor_histograms=unflatten_descriptor(avg_descs[id_image]),
            descriptor_name=descs_names[id_image],
            show_plot=False,
        )

    plt.show()
