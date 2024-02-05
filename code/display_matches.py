# Testing pipeline after Blender script execution
import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../matching")
from saving import load_matches, load_keypoints

# TODO: Filter matches, not keypoints

if __name__ == "__main__":
    im_folder = "../data/blender/rocks/"
    photo_name = "rock_1"
    im_names = ["rock_1_left", "rock_1_right"]

    im_1 = cv.imread(im_folder + "left.png", cv.IMREAD_GRAYSCALE)
    im_2 = cv.imread(im_folder + "right.png", cv.IMREAD_GRAYSCALE)
    ims = [im_1, im_2]

    storage_folders = ["computed_descriptors", "computed_distances", "computed_matches"]

    y_starts = [400, 400]
    y_lengths = [5, 5]
    x_starts = [800, 800]
    x_lengths = [5, 5]

    # load all computed objects
    matches_filename_prefix = f"{photo_name}_y_{y_starts[0]}_{y_starts[1]}_{y_lengths[0]}_{y_lengths[1]}_{x_starts[0]}_{x_starts[1]}_{x_lengths[0]}_{x_lengths[1]}"
    unfiltered_filename_prefixes = [
        f"{im_names[id_image]}_y_{y_starts[id_image]}_{y_lengths[id_image]}_x_{x_starts[id_image]}_{x_lengths[id_image]}"
        for id_image in range(2)
    ]

    # load unfiltered keypoints coordinates
    unfiltered_kps_pos_filenames = [
        f"{storage_folders[0]}/{unfiltered_filename_prefixes[id_image]}_coords.npy"
        for id_image in range(2)
    ]
    unfiltered_kps_pos = [
        np.load(unfiltered_kps_pos_filenames[id_image]) for id_image in range(2)
    ]

    # display unfiltered keypoints
    for id_image in range(2):
        pos_xs = unfiltered_kps_pos[id_image][:, 0]
        pos_ys = unfiltered_kps_pos[id_image][:, 1]

        plt.figure(figsize=(10, 10))
        plt.imshow(ims[id_image], cmap="gray")
        plt.scatter(pos_xs, pos_ys, c="r", s=10)
        plt.axis("off")
        plt.show()

    # load filtered keypoints, matches and index of good matches
    matched_kps_filenames = [
        f"{storage_folders[2]}/{matches_filename_prefix}_kp_{id_image}.txt"
        for id_image in range(2)
    ]
    matched_kps = [
        load_keypoints(kps_filename) for kps_filename in matched_kps_filenames
    ]

    matches_idxs_filename = f"{storage_folders[2]}/{matches_filename_prefix}_idxs.npy"
    matches_idxs = np.load(matches_idxs_filename)

    matches_filename = f"{storage_folders[2]}/{matches_filename_prefix}_matches.txt"
    matches = load_matches(matches_filename)

    good_matches = [matches[i] for i in matches_idxs]

    for id_image in range(2):
        print(f"number of keypoints in image {id_image}", len(matched_kps[id_image]))
        print(
            f"some keypoints positions in image {id_image}", matched_kps[id_image][10]
        )
    print("number of matches", len(matches))
    print("good matches", good_matches)

    # draw matches
    matches_img = cv.drawMatchesKnn(
        # Warning : the number of matches to draw is not specified here
        img1=im_1,
        keypoints1=matched_kps[0],
        img2=im_2,
        keypoints2=matched_kps[1],
        matches1to2=good_matches,
        outImg=None,
        singlePointColor=(255, 0, 0),
    )

    plt.figure(figsize=(10, 5))
    plt.imshow(matches_img)
    plt.axis("off")
    plt.show()
