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
        np.load(f"{descrip_path}/{kp_coords_filenames[id_image]}{sift_suffix}.npy")
        for id_image in range(2)
    ]

    # load unfiltered keypoints, matches and index of good matches
    kps = [
        load_keypoints(f"{matches_path}/{kp_filenames[id_image]}{sift_suffix}")
        for id_image in range(2)
    ]
    matches = load_matches(f"{matches_path}/{matches_filename}{sift_suffix}")

    # Load matches filtered by blender
    correct_matches_idxs = np.load(
        f"{matches_path}/{correct_matches_idxs_filename}{sift_suffix}.npy"
    )

    # filter good matches according to blender
    good_matches = [matches[i] for i in correct_matches_idxs]

    # print general info about proportions of keypoints and matches

    for id_image in range(2):
        print(f"number of sift keypoints in image {id_image}", len(kps[id_image]))
    print("number of unfiltered sift matches", len(matches))
    print(
        f"number of good sift matches at a precision of {epsilon} pixels: ",
        len(good_matches),
    )

    # look at some matches

    chosen_matches_idx = [1]
    for match_idx in chosen_matches_idx:
        # display 1 match, object here is not DMatch, but a couple of DMatch, as Sift returns
        # we get here only the Dmatch
        chosen_Dmatch = good_matches[match_idx][0]

        # display the match
        dm.display_match(
            ims,
            chosen_Dmatch,
            kps_coords,
            show_plot=False,
            # save_path=filtered_kp_path,  # comment or pass None to not save the image
            filename_prefix=f"{correct_match_filename_prefix}{sift_suffix}",
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
