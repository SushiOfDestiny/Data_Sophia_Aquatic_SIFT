# Testing pipeline after Blender script execution
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
    im_names=None,
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
    title = f"Match between {im_names[0]} and {im_names[1]}, with distance {dmatch.distance:.2f} {comment_dist_type}, \n at coordinates {np.round(matched_kps_pos[0])} and {np.round(matched_kps_pos[1])}, \n with precision threshold {epsilon} pixels"
    plt.suptitle(title)

    if save_path is not None and filename_prefix is not None:
        filename_suffix = f"_{matched_kps_pos[0][0]}_{matched_kps_pos[0][1]}_{matched_kps_pos[1][0]}_{matched_kps_pos[1][1]}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f"{save_path}/{filename_prefix}_{filename_suffix}.png", dpi=dpi)

    if show_plot:
        plt.show()


def print_general_kp_matches_infos(ims, kps, matches, good_matches, epsilon):
    """print general infos and percentage of keypoints and good matches among the pixels and matches."""
    for id_image in range(2):
        print(f"number of pixels in image {id_image}", ims[id_image].size)
        print(f"number of sift keypoints in image {id_image}", len(kps[id_image]))
        print(
            f"percentage of sift keypoints in image {id_image}",
            len(kps[id_image]) / ims[id_image].size * 100.0,
        )
    print("number of unfiltered sift matches", len(matches))
    print(
        f"number of good sift matches at a precision of {epsilon} pixels: ",
        len(good_matches),
    )
    print(
        f"Percentage of good matches within matches: {len(good_matches) / len(matches) * 100.0}"
    )
    print(
        f"Percentage of good matches within pixels in subimage 1: {len(good_matches) / ims[0].size * 100.0}"
    )


def print_distance_infos(matches_list):
    """
    Print some information about the distances of the matches.
    matches_list: list of SIFT matches (pairs of DMatch objects)
    print the minimal, maximal and mean distances and standard deviation of the distances
    """
    distances = np.array([match[0].distance for match in matches_list])
    print(f"Minimal distance: {np.min(distances)}")
    print(f"Maximal distance: {np.max(distances)}")
    print(f"Mean distance: {np.mean(distances)}")
    print(f"Standard deviation of the distances: {np.std(distances)}")


def compute_good_and_bad_matches(matches, good_matches_kps_idx):
    """
    Compute the good and bad matches from the list of matches and the index of the good matches
    return the good and bad matches as sift matches (pairs of DMatch objects)
    """
    good_matches = [matches[i] for i in good_matches_kps_idx]
    bad_matches_idx = np.setdiff1d(np.arange(len(matches)), good_matches_kps_idx)
    bad_matches = [matches[i] for i in bad_matches_idx]
    return good_matches, bad_matches


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

    # load cropped float images
    cropped_float_ims = [
        np.load(
            f"{original_imgs_path_prefix}/{cropped_float_ims_filenames[id_image]}.npy"
        )
        for id_image in range(2)
    ]

    # show cropped float images
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    for id_image in range(2):
        axs[id_image].imshow(cropped_float_ims[id_image], cmap="gray")

    plt.show()

    # load all computed objects

    # load unfiltered keypoints coordinates
    kps_coords = [
        np.load(f"{descrip_path}/{kp_coords_filenames[id_image]}.npy")
        for id_image in range(2)
    ]
    descs = [
        np.load(
            f"{descrip_path}/{descrip_filenames[id_image]}.npy",
        )
        for id_image in range(2)
    ]

    # load unfiltered keypoints, matches and index of good matches
    kps = [
        load_keypoints(f"{matches_path}/{kp_filenames[id_image]}.txt")
        for id_image in range(2)
    ]
    matches = load_matches(f"{matches_path}/{matches_filename}.txt")

    # Load matches filtered by blender
    correct_matches_idxs = np.load(
        f"{matches_path}/{correct_matches_idxs_filename}.npy"
    )

    # filter good matches and bad matches according to blender
    good_matches, bad_matches = compute_good_and_bad_matches(
        matches, correct_matches_idxs
    )

    # print general info about proportions of keypoints and matches
    print_general_kp_matches_infos(float_ims, kps, matches, good_matches, epsilon)
    

    # look at some matches

    chosen_matches_idx = [0, 1]
    for match_idx in chosen_matches_idx:
        # display 1 match, object here is not DMatch, but a couple of DMatch, as Sift returns
        # we get here only the Dmatch
        chosen_Dmatch = good_matches[match_idx][0]

        # display the match
        display_match(
            ims,
            chosen_Dmatch,
            kps_coords,
            show_plot=True,
            # save_path=filtered_kp_path,  # comment or pass None to not save the image
            filename_prefix=correct_match_filename_prefix,
            dpi=800,
            im_names=im_names,
        )

        # display topological properties
        chosen_kps = [kps[0][chosen_Dmatch.queryIdx], kps[1][chosen_Dmatch.trainIdx]]
        # vh.topological_visualization_pipeline(
        #     kps=chosen_kps,
        #     uint_ims=ims,
        #     float_ims=float_ims,
        #     zoom_radius=20,
        #     show_directions=False,
        #     show_gradients=False,
        #     show_plot=False,
        # )

        # display the descriptor of the point in the 2 images
        # for id_image in range(2):
        #     visu_desc.display_descriptor(
        #         descriptor_histograms=unflatten_descriptor(
        #             descs[id_image][chosen_Dmatch.queryIdx],
        #             nb_bins=nb_bins,
        #             nb_angular_bins=nb_angular_bins,
        #         ),
        #         descriptor_name=f"Descriptor of the match {match_idx} in {im_names[id_image]}",
        #         show_plot=False,
        #     )

    # look at the average descriptors of the good matches
    good_matches_kps = [
        [kps_coords[0][match[0].queryIdx] for match in good_matches],
        [kps_coords[1][match[0].trainIdx] for match in good_matches],
    ]
    good_matches_kps_idx = [
        np.array(
            [
                (kp[1] - y_starts[id_image]) * x_lengths[id_image]
                + (kp[0] - x_starts[id_image])
                for kp in good_matches_kps[id_image]
            ]
        )
        for id_image in range(2)
    ]

    good_descs_ims = [
        descs[id_image][good_matches_kps_idx[id_image]] for id_image in range(2)
    ]
    avg_good_descs = [
        np.mean(good_descs_ims[id_image], axis=0) for id_image in range(2)
    ]

    good_descs_names = [
        f"Averaged descriptor of good matches for {im_names[id_image]}\n with nb_bins={nb_bins}, bin_radius={bin_radius}, delta_angle={delta_angle} and sigma={sigma}"
        for id_image in range(2)
    ]

    # look at averaged bad descriptor

    bad_descs = [
        descs[id_image][
            np.setdiff1d(
                np.arange(np.shape(descs[id_image])[0]), good_matches_kps_idx[id_image]
            )
        ]
        for id_image in range(2)
    ]

    avg_bad_descs = [np.mean(bad_descs[id_image], axis=0) for id_image in range(2)]

    bad_descs_names = [
        f"Averaged descriptor of bad matches for {im_names[id_image]}\n with nb_bins={nb_bins}, bin_radius={bin_radius}, delta_angle={delta_angle} and sigma={sigma}"
        for id_image in range(2)
    ]

    avg_descs = [avg_good_descs, avg_bad_descs]
    descs_names = [good_descs_names, bad_descs_names]

    # for id_desc in range(2):
    #     for id_image in range(2):
    #         visu_desc.display_descriptor(
    #             descriptor_histograms=unflatten_descriptor(
    #                 avg_descs[id_desc][id_image],
    #                 nb_bins=nb_bins,
    #                 nb_angular_bins=nb_angular_bins,
    #             ),
    #             descriptor_name=descs_names[id_desc][id_image],
    #             show_plot=False,
    #         )

    # visu_desc.display_descriptor(
    #         descriptor_histograms=unflatten_descriptor(kps_coords[match_idx])
    # )

    # plt.show()

    # look at the distances of the good and bad matches
    print("Statistics about the distances of the good matches")
    print_distance_infos(good_matches)

    print("Statistics about the distances of the bad matches")
    print_distance_infos(bad_matches)

    # look at filtered keypoints on image
    if use_filt:
        # plot the filtered pixels on each subimage side by side
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        for id_image in range(2):
            ax[id_image].imshow(float_ims[id_image], cmap="gray")
            ax[id_image].scatter(
                kps_coords[id_image][:, 0],
                kps_coords[id_image][:, 1],
                c="r",
                s=2,
            )
            ax[id_image].set_title(f"Filtered pixels in subimage {id_image}")
        plt.show()
