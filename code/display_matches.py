# Testing pipeline after Blender script execution
import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from descriptor import unflatten_descriptor
from visu_descriptor import display_descriptor

sys.path.append("../matching")
from saving import load_matches, load_keypoints


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
    title = f"Match between image 1 and image 2, with distance {dmatch.distance} {comment_dist_type}, \n at coordinates {np.round(matched_kps_pos[0])} and {np.round(matched_kps_pos[1])}, \n with precision threshold {epsilon} pixels"
    plt.suptitle(title)

    if save_path is not None and filename_prefix is not None:
        filename_suffix = f"_{matched_kps_pos[0][0]}_{matched_kps_pos[0][1]}_{matched_kps_pos[1][0]}_{matched_kps_pos[1][1]}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f"{save_path}/{filename_prefix}_{filename_suffix}.png", dpi=dpi)

    if show_plot:
        plt.show()


if __name__ == "__main__":
    im_folder = "../data/blender/rocks/"
    photo_name = "rock_1"
    im_names = ["rock_1_left", "rock_1_right"]

    im_1 = cv.imread(im_folder + "left.png", cv.IMREAD_GRAYSCALE)
    im_2 = cv.imread(im_folder + "right.png", cv.IMREAD_GRAYSCALE)
    ims = [im_1, im_2]

    # set the coordinates of the subimages
    y_starts = [386, 459]
    y_lengths = [10, 10]
    x_starts = [803, 806]
    x_lengths = [20, 20]
    # define distance type suffix
    distance_type = "min"
    prefixes_extension = "" if distance_type == "all" else "_min"

    # redefine the threshold used
    epsilon = 1

    # load all computed objects
    matches_filename_prefix = f"{photo_name}_y_{y_starts[0]}_{y_starts[1]}_{y_lengths[0]}_{y_lengths[1]}_x_{x_starts[0]}_{x_starts[1]}_{x_lengths[0]}_{x_lengths[1]}{prefixes_extension}"
    unfiltered_filename_prefixes = [
        f"{im_names[id_image]}_y_{y_starts[id_image]}_{y_lengths[id_image]}_x_{x_starts[id_image]}_{x_lengths[id_image]}"
        for id_image in range(2)
    ]

    # load unfiltered keypoints coordinates
    kps_coords_filenames = [
        f"computed_descriptors/{unfiltered_filename_prefixes[id_image]}_coords.npy"
        for id_image in range(2)
    ]
    kps_coords = [np.load(kps_coords_filenames[id_image]) for id_image in range(2)]
    descs = [np.load(f"computed_descriptors/{unfiltered_filename_prefixes[id_image]}_descs.npy") for id_image in range(2)]

    # load filtered keypoints, matches and index of good matches
    kps = [
        load_keypoints(f"computed_matches/{matches_filename_prefix}_kp_{id_image}.txt")
        for id_image in range(2)
    ]

    matches_idxs_filename = (
        f"computed_matches/{matches_filename_prefix}_correct_idxs.npy"
    )
    matches_idxs = np.load(matches_idxs_filename)

    matches_filename = f"computed_matches/{matches_filename_prefix}_matches.txt"
    matches = load_matches(matches_filename)

    # filter good matches according to blender
    good_matches = [matches[i] for i in matches_idxs]

    for id_image in range(2):
        print(f"number of keypoints in image {id_image}", len(kps[id_image]))
        # print(f"some keypoints positions in image {id_image}", kps_coords[id_image][10])
    print("number of unfiltered matches", len(matches))
    print("nb good matches", len(good_matches))

    # draw matches
    matches_img = cv.drawMatchesKnn(
        # Warning : the number of matches to draw is not specified here
        img1=im_1,
        keypoints1=kps[0],
        img2=im_2,
        keypoints2=kps[1],
        matches1to2=good_matches,
        outImg=None,
        singlePointColor=(255, 0, 0),
    )

    # display 1 match, object here is not DMatch, but a couple of DMatch, as Sift returns
    match_idx = 0
    # we get here only the Dmatch
    chosen_Dmatch = good_matches[match_idx][0]

    display_match(
        ims,
        chosen_Dmatch,
        kps_coords,
        show_plot=True,
        save_path="filtered_keypoints",
        filename_prefix=f"{matches_filename_prefix}_correct_match",
        dpi=800,
    )
    
    # pabo
    good_matches_kps_1 = [kps_coords[0][dmatch[0].queryIdx] for dmatch in good_matches]
    good_matches_kps_2 = [kps_coords[1][dmatch[0].trainIdx] for dmatch in good_matches]
    good_matches_kps_1_idxs = np.array([(kp[1] - y_starts[0])*x_lengths[0]+(kp[0] - x_starts[0]) for kp in good_matches_kps_1])
    good_matches_kps_2_idxs = np.array([(kp[1] - y_starts[1])*x_lengths[0]+(kp[0] - x_starts[1]) for kp in good_matches_kps_2])
    good_descs_1 = descs[0][good_matches_kps_1_idxs]
    good_descs_2 = descs[1][good_matches_kps_2_idxs]
    
    avg_desc_1 = np.mean(good_descs_1, axis=0)
    avg_desc_2 = np.mean(good_descs_2, axis=0)

    display_descriptor(unflatten_descriptor(avg_desc_1, nb_bins=1, nb_angular_bins=(int(360.0/5.0)+1)))
