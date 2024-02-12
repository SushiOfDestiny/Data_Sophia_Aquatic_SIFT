import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from computation_pipeline_hyper_params import *
from filenames_creation import *


from general_pipeline_before_blender import create_logger

sys.path.append("../matching")
from saving import (
    save_keypoints,
    save_matches,
    save_kp_pairs_to_arr,
)
from matching import get_keypoint_pairs, draw_good_keypoints


def translate_kp_coords(kp, y_start, x_start):
    """
    Translate the coordinates of a keypoint.
    Return a new translated keypoint, because postion is an immutable tuple.
    """
    trans_kp = cv.KeyPoint(
        x=kp.pt[0] + x_start,
        y=kp.pt[1] + y_start,
        size=kp.size,
    )
    return trans_kp


if __name__ == "__main__":
    ims = [
        cv.imread(
            f"{original_imgs_path_prefix}/{im_names[i]}.{im_ext}", cv.IMREAD_GRAYSCALE
        )
        for i in range(2)
    ]

    # Load cropped images
    cropped_ims = [
        cv.imread(
            f"{original_imgs_path_prefix}/{cropped_ims_filenames[id_image]}.{im_ext}",
            cv.IMREAD_GRAYSCALE,
        )
        for id_image in range(2)
    ]

    # compute sift keypoints and matches
    method_post = "lowe"
    contrastThreshold = 0.00
    edgeThreshold = 0
    SIFTsigma = 0.1
    distanceThreshold = 1e9

    print("SIFT parameters: ")
    print(f"contrastThreshold: {contrastThreshold}")
    print(f"edgeThreshold: {edgeThreshold}")
    print(f"SIFTsigma: {SIFTsigma}")
    print(f"distanceThreshold: {distanceThreshold}")

    sift_kp_pairs, sift_matches, sift_kp1, sift_kp2 = get_keypoint_pairs(
        cropped_ims[0],
        cropped_ims[1],
        method_post=method_post,
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        SIFTsigma=SIFTsigma,
        distanceThreshold=distanceThreshold,
    )

    sift_kps = [sift_kp1, sift_kp2]

    # # draw the matches
    # draw_good_keypoints(
    #     cropped_ims[0], cropped_ims[1], sift_kp1, sift_kp2, sift_matches, 5
    # )

    # translate coords of keypoints from the subimage frame to the whole image frame
    trans_sift_kps = [None, None]
    for id_image in range(2):
        trans_sift_kps[id_image] = [
            translate_kp_coords(
                sift_kps[id_image][id_kp], y_starts[id_image], x_starts[id_image]
            )
            for id_kp in range(len(sift_kps[id_image]))
        ]
    # translate also coords in the sift_kp_pairs
    trans_sift_kp_pairs = [None for _ in range(len(sift_kp_pairs))]
    for id_pair in range(len(sift_kp_pairs)):
        for id_image in range(2):
            trans_sift_kp_pairs[id_pair] = [
                translate_kp_coords(
                    sift_kp_pairs[id_pair][id_image],
                    y_starts[id_image],
                    x_starts[id_image],
                )
                for id_image in range(2)
            ]

    # # draw the translated matches
    # draw_good_keypoints(
    #     ims[0], ims[1], trans_sift_kps[0], trans_sift_kps[1], sift_matches, 5
    # )

    # compute keypoints int pixels coordinates as list of 2 numpy arrays
    trans_sift_kps_coords = [
        np.zeros(shape=(len(trans_sift_kps[id_image]), 2), dtype=np.int32)
        for id_image in range(2)
    ]

    for id_image in range(2):
        for id_kp in range(len(trans_sift_kps[id_image])):
            trans_sift_kps_coords[id_image][id_kp, :] = np.array(
                [
                    int(np.round(trans_sift_kps[id_image][id_kp].pt[0])),
                    int(np.round(trans_sift_kps[id_image][id_kp].pt[1])),
                ]
            )

    # # display some coords of kpts
    # fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    # for id_image in range(2):
    #     axs[id_image].scatter(
    #         trans_sift_kps_coords[id_image][:, 0], trans_sift_kps_coords[id_image][:, 1]
    #     )
    #     axs[id_image].imshow(ims[id_image], cmap="gray")
    # plt.show()

    # save trans sift coords of kps
    for id_image in range(2):
        np.save(
            f"{descrip_path}/{kp_coords_filenames[id_image]}{sift_suffix}",
            trans_sift_kps_coords[id_image],
        )
    print("finished saving translated sift kp coords")

    # save the matches and keypoints using function from matching/saving.py
    save_kp_pairs_to_arr(
        trans_sift_kp_pairs,
        f"{matches_path}/{kp_pairs_filename}{sift_suffix}",
    )
    print("finished saving translated sift kp_pairs")

    save_matches(
        sift_matches,
        f"{matches_path}/{matches_filename}{sift_suffix}",
    )
    print("finished saving sift matches")

    for id_image in range(2):
        save_keypoints(
            trans_sift_kps[id_image],
            f"{matches_path}/{kp_filenames[id_image]}{sift_suffix}",
        )
    print("finished saving translated sift keypoints")
