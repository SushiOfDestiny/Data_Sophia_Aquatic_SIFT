# Testing pipeline after Blender script execution
import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../matching")
from saving import load_matches, load_keypoints

from computation_pipeline_hyper_params import *

from filenames_creation import *


def display_match(
    ims,
    dmatch,
    kps_coords,
    show_plot=False,
    save_path="filtered_keypoints",
    filename_prefix=None,
    dpi=800,
    epsilon=1,
    distance_type="min",
):
    """
    Plot a match between the 2 images
    ims: list of 2 images, each being a numpy array
    dmatch: DMatch object
    kps_coords: list of 2 numpy arrays of shape (n, 2), each containing the coordinates of the keypoints of the image, coordinates are (x, y)
    show_plot: boolean, whether to display the plot or not
    save_path: string, path to the directory where the image should be saved
    filename_prefix: string, beginning of name of the file to save the image as
    dpi: int, resolution of the saved image in dots per inch
    """
    matched_kps_pos = (
        kps_coords[0][dmatch.queryIdx],
        kps_coords[1][dmatch.trainIdx],
    )

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    for id_image in range(2):
        axs[id_image].imshow(ims[id_image], cmap="gray")
        axs[id_image].scatter(
            matched_kps_pos[id_image][0], matched_kps_pos[id_image][1], c="r", s=10
        )

    # add title
    comment_dist_type = "minimal for pixel1" if distance_type == "min" else ""
    title = f"Match between image 1 and image 2, with distance {dmatch.distance:.2f} {comment_dist_type}, \n at coordinates {np.round(matched_kps_pos[0])} and {np.round(matched_kps_pos[1])}, \n with precision threshold {epsilon} pixels"
    plt.suptitle(title)

    if save_path is not None and filename_prefix is not None:
        filename_suffix = f"_{matched_kps_pos[0][0]}_{matched_kps_pos[0][1]}_{matched_kps_pos[1][0]}_{matched_kps_pos[1][1]}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f"{save_path}/{filename_prefix}_{filename_suffix}.png", dpi=dpi)

    if show_plot:
        plt.show()


if __name__ == "__main__":

    ims = [
        cv.imread(f"{img_folder}/{im_names[id_image]}.{im_ext}", cv.IMREAD_GRAYSCALE)
        for id_image in range(2)
    ]

    # load all computed objects

    # load unfiltered keypoints coordinates
    kps_coords = [np.load(kp_coords_filenames[id_image]) for id_image in range(2)]
    descs = [
        np.load(
            descrip_filenames[id_image],
        )
        for id_image in range(2)
    ]

    # load unfiltered keypoints, matches and index of good matches
    kps = [load_keypoints(kp_filenames[id_image]) for id_image in range(2)]

    matches_idxs = np.load(f"{matches_path}/{correct_matches_idxs_filename}.npy")

    matches = load_matches(f"{matches_path}/{matches_filename}")

    # filter good matches according to blender
    good_matches = [matches[i] for i in matches_idxs]

    for id_image in range(2):
        print(f"number of keypoints in image {id_image}", len(kps[id_image]))
        # print(f"some keypoints positions in image {id_image}", kps_coords[id_image][10])
    print("number of unfiltered matches", len(matches))
    print("nb good matches", len(good_matches))

    # look at some matches
    chosen_matches_idx = [1, 2, 3]
    for match_idx in chosen_matches_idx:
        # display 1 match, object here is not DMatch, but a couple of DMatch, as Sift returns
        # we get here only the Dmatch
        chosen_Dmatch = good_matches[match_idx][0]

    display_match(
        ims,
        chosen_Dmatch,
        kps_coords,
        show_plot=True,
        save_path="filtered_keypoints",
        filename_prefix=correct_match_filename_prefix,
        dpi=800,
    )

    # pabo
    good_matches_kps_1 = [kps_coords[0][dmatch.queryIdx] for dmatch in good_matches]
    good_matches_kps_2 = [kps_coords[1][dmatch.trainIdx] for dmatch in good_matches]
    good_matches_kps_1_idxs = np.array(
        [(kp[1] - y_starts[0]) * x_lengths[0] + (kp[0] - x_starts[0])]
        for kp in good_matches_kps_1
    )
    good_matches_kps_2_idxs = np.array(
        [(kp[1] - y_starts[1]) * x_lengths[0] + (kp[0] - x_starts[1])]
        for kp in good_matches_kps_2
    )
    good_descs_1 = descs[0][good_matches_kps_1_idxs]
    good_descs_2 = descs[1][good_matches_kps_2_idxs]

    avg_desc_1 = np.mean(good_descs_1, axis=0)
    avg_desc_2 = np.mean(good_descs_2, axis=0)
