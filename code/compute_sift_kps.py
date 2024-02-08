import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from computation_pipeline_hyper_params import *
from filenames_creation import *

sys.path.append("../matching")
from saving import (
    save_keypoints,
    save_matches,
    save_kp_pairs_to_arr,
)
from matching import get_keypoint_pairs, draw_good_keypoints

if __name__ == "__main__":
    # Load images
    ims = [
        cv.imread(
            f"{original_imgs_path_prefix}/{im_names[i]}.{im_ext}", cv.IMREAD_GRAYSCALE
        )
        for i in range(2)
    ]

    # # Crop images
    # cropped_ims = [
    #     ims[id_image][
    #         y_starts[id_image] : y_starts[id_image] + y_lengths[id_image],
    #         x_starts[id_image] : x_starts[id_image] + x_lengths[id_image],
    #     ]
    #     for id_image in range(2)
    # ]
    cropped_ims = ims

    # compute sift keypoints and matches
    sift_kp_pairs, sift_matches, sift_kp1, sift_kp2 = get_keypoint_pairs(
        cropped_ims[0], cropped_ims[1], method_post="lowe"
    )
    sift_kps = [sift_kp1, sift_kp2]
    # compute keypoints coordinates as list of 2 numpy arrays
    sift_kps_coords = [
        np.zeros(shape=(len(sift_kps[id_image]), 2), dtype=np.int32) for id_image in range(2)
    ]
    for id_image in range(2):
        for id_kp in range(len(sift_kps[id_image])):
            sift_kps_coords[id_image][id_kp, :] = np.array(
                [
                    int(np.round(sift_kps[id_image][id_kp].pt[0])),
                    int(np.round(sift_kps[id_image][id_kp].pt[1])),
                ]
            )

    # draw keypoints
    # draw_good_keypoints(
    #     cropped_ims[0],
    #     cropped_ims[1],
    #     sift_kps[0],
    #     sift_kps[1],
    #     sift_matches,
    #     20,
    # )

    # save sift coords of kps
    for id_image in range(2):
        np.save(
            f"{descrip_path}/{kp_coords_filenames[id_image]}{sift_suffix}",
            sift_kps_coords[id_image],
        )

    # save the matches and keypoints using function from matching/saving.py
    save_kp_pairs_to_arr(
        sift_kp_pairs,
        f"{matches_path}/{kp_pairs_filename}{sift_suffix}",
    )
    print("finished saving sift kp_pairs")

    save_matches(
        sift_matches,
        f"{matches_path}/{matches_filename}{sift_suffix}",
    )
    print("finished saving sift matches")

    for id_image in range(2):
        save_keypoints(
            sift_kps[id_image],
            f"{matches_path}/{kp_filenames[id_image]}{sift_suffix}",
        )
    print("finished saving sift keypoints")
