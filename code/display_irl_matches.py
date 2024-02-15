# Goal of this script is to visualise some matches on real images
import os
import sys

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import descriptor as desc
import visu_hessian as vh
import visu_descriptor as visu_desc

sys.path.append("../matching")
from saving import load_matches, load_keypoints

from computation_pipeline_hyper_params import *

from filenames_creation import *

import display_matches as dm
import compute_desc_img as cp_desc
import descriptor as desc
from matplotlib.ticker import PercentFormatter

import compute_desc_img as cp_desc


if __name__ == "__main__":

    plot_hist = False

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

    # load all computed objects

    # load unfiltered keypoints coordinates
    kps_coords = [
        np.load(f"{descrip_path}/{kp_coords_filenames[id_image]}.npy")
        for id_image in range(2)
    ]

    # load unfiltered keypoints, matches and index of good matches
    kps = [
        load_keypoints(f"{matches_path}/{kp_filenames[id_image]}.txt")
        for id_image in range(2)
    ]
    matches = load_matches(f"{matches_path}/{matches_filename}.txt")

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

    # look at statistics about the distances of the matches
    print("Statistics about the distances of the matches")
    dm.print_distance_infos(matches)

    # Sort matches by distance
    minimal_matches_idx = np.argsort([match[0].distance for match in matches])
    minimal_matches = [matches[i] for i in minimal_matches_idx]

    # display random minimal matches
    max_nb_matches_to_display = len(matches)
    nb_rd_unfiltered_matches_to_display = min(
        len(minimal_matches), max_nb_matches_to_display
    )
    rd_idx_unfiltered = np.random.choice(
        len(matches), nb_rd_unfiltered_matches_to_display
    )
    rd_matches_unfiltered = [matches[i] for i in rd_idx_unfiltered]
    dm.display_matches(
        ims,
        rd_matches_unfiltered,
        kps_coords,
        show_plot=False,
        im_names=im_names,
        plot_title_prefix=f"Random minimal unfiltered by blender {sift_radical[1:] if use_sift else ''} matches",
    )

    plt.show()

    if not use_sift:

        if plot_hist:
            overall_features_ims = [
                desc.compute_features_overall_abs(
                    float_ims[id_image], border_size=border_size
                )
                for id_image in range(2)
            ]
            for id_image in range(2):
                mean_abs_curvs = cp_desc.compute_mean_abs_curv_arr(
                    overall_features_ims[id_image][1]
                )
                plt.figure()
                hist = plt.hist(
                    x=mean_abs_curvs,
                    nbins=100,
                    weights=np.ones(mean_abs_curvs) / len(mean_abs_curvs),
                )
                plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
                plt.xlabel("Moyenne des valeurs absolues des courbures")
                plt.ylabel("Pourcentage")

        # for id_image in range(2):
        #     visu_desc.display_descriptor(
        #         descriptor_histograms=desc.unflatten_descriptor(avg_descs[id_image]),
        #         descriptor_name=descs_names[id_image],
        #         show_plot=False,
        #     )

        plt.show()

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
                ax[id_image].set_title(f"Prefiltered pixels in subimage {id_image}")
            plt.show()

            # Look at information about the curvatures and gradients of the good and bad keypoints
            print(f"feat computation beginning for both images")
            overall_features_ims = [
                desc.compute_features_overall_abs(
                    float_ims[id_image], border_size=border_size
                )
                for id_image in range(2)
            ]

            # look at the mean absolute curvatures
            mean_abs_curvs_ims = [
                cp_desc.compute_mean_abs_curv_arr(overall_features_ims[id_image][1])
                for id_image in range(2)
            ]

            # look at mask of prefiltered pixels

            mask_arrays = [None, None]
            if filt_type is None or filt_type == "mean":
                for id_image in range(2):

                    mask_arrays[id_image], mean_abs_curvs, y_slice, x_slice = (
                        cp_desc.filter_by_mean_abs_curv(
                            float_ims[id_image],
                            overall_features_ims[id_image][1],
                            y_starts[id_image],
                            y_lengths[id_image],
                            x_starts[id_image],
                            x_lengths[id_image],
                            percentile,
                        )
                    )

            elif filt_type == "std":
                for id_image in range(2):
                    mask_arrays[id_image], neighborhood_stds, y_slice, x_slice = (
                        cp_desc.filter_by_std_neighbor_curv(
                            float_ims[id_image],
                            overall_features_ims[id_image][1],
                            y_starts[id_image],
                            y_lengths[id_image],
                            x_starts[id_image],
                            x_lengths[id_image],
                            percentile,
                            neighborhood_radius=neighborhood_radius,
                        )
                    )
                    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
                    axs[0].imshow(
                        float_ims[id_image][
                            y_starts[id_image] : y_starts[id_image]
                            + y_lengths[id_image],
                            x_starts[id_image] : x_starts[id_image]
                            + x_lengths[id_image],
                        ],
                        cmap="gray",
                    )
                    axs[1].imshow(neighborhood_stds, cmap="magma")
                    axs[0].set_title(
                        f"Colormap of filtered pixels by {filt_type} for image {id_image}"
                    )

                    dm.save_fig_pkl(
                        fig,
                        path=descrip_path,
                        filename=f"{descrip_filename_prefixes[id_image]}_filter_colormap",
                    )

                    plt.show()

            # display the filtered pixels
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            for id_image in range(2):
                axs[id_image].imshow(float_ims[id_image], cmap="gray")
                axs[id_image].imshow(mask_arrays[id_image], cmap="jet", alpha=0.5)
                axs[id_image].set_title(
                    f"Filtered pixels by {filt_type} curvature in subimage {id_image}, with percentile {percentile}"
                )

            # save the plot
            dm.save_fig_pkl(
                fig,
                path=descrip_path,
                filename=f"{descrip_filename_prefixes[id_image]}_filtered_pixels",
            )

            plt.show()

            # Load descriptors
            descs = [
                np.load(
                    f"{descrip_path}/{descrip_filenames[id_image]}.npy",
                )
                for id_image in range(2)
            ]

            descs_ims = [
                descs[id_image][matches_kps_idx[id_image]] for id_image in range(2)
            ]

            # Look at the averaged descriptor
            avg_descs = [np.mean(descs_ims[id_image], axis=0) for id_image in range(2)]

            descs_names = [
                f"Averaged descriptor of matches for {im_names[id_image]}\n with nb_bins={nb_bins}, bin_radius={bin_radius}, delta_angle={delta_angle} and sigma={sigma}"
                for id_image in range(2)
            ]
