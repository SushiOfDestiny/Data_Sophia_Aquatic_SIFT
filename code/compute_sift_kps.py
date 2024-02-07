import sys
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

    # save sift matches and keypoints
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
