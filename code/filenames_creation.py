import numpy as np
import os

from computation_pipeline_hyper_params import *

# Goal: create the filenames for all objects save during the computation pipeline

# CONVENTION: all paths variables end with "_path"

# Create multiple suffix to distinguish between pipelines
# Create filename suffix for sift kps, pairs and matches
sift_suffix = "_sift"
sift_radical = "_sift" if use_sift else ""
cropped_sift_radical = "" if not use_sift else sift_radical[1:]

# create suffix for filtered points computation
filt_suffix = "_filt"
filt_radical = f"_filt_{percentile}" if use_filt else ""
# create suffix for type of distance
dist_type_suffix = "" if distance_type == "all" else "_min"

# Create filename radical for irl images
irl_radical = "_irl" if img_folder[:3] == "irl" else ""


# Create filename where to load the original images
original_imgs_path_prefix = f"{relative_path}/{img_folder}"

# Create the filename for the blurred images
blurred_imgs_filename = (
    f"{original_imgs_path_prefix}/blurred_{photo_name}_bsig{int(blur_sigma)}"
)


# Create suffix for descriptor parameters
raw_descrip_suffix = (
    f"_nbins_{nb_bins}_brad_{bin_radius}_nangbins_{nb_angular_bins}_sig{int(sigma)}"
)
descrip_suffix = f"{raw_descrip_suffix}{sift_radical}{filt_radical}"

# Define path to save the descriptors and coordinates
descrip_path = "computed_descriptors"
# Create filename for the descriptors and coordinates
raw_descrip_prefixes = [
    f"{photo_name}_{im_names[id_image]}_y_{y_starts[id_image]}_{y_lengths[id_image]}_x_{x_starts[id_image]}_{x_lengths[id_image]}"
    for id_image in range(2)
]
descrip_filename_prefixes = [
    f"{raw_descrip_prefixes[id_image]}{descrip_suffix}" for id_image in range(2)
]
descrip_filenames = descrip_filename_prefixes

# Create filename for the keypoints coordinates
kp_coords_filenames = [
    f"{descrip_filename_prefixes[id_image]}_coords" for id_image in range(2)
]


# Define path to save the distances and matches
dist_path = "computed_distances"
# Create filename for the distances and matches
raw_dist_prefix = f"{photo_name}_y_{y_starts[0]}_{y_starts[1]}_{y_lengths[0]}_{y_lengths[1]}_x_{x_starts[0]}_{x_starts[1]}_{x_lengths[0]}_{x_lengths[1]}"
dist_filename_prefix = f"{raw_dist_prefix}{descrip_suffix}{dist_type_suffix}"
dist_filename = f"{dist_filename_prefix}_dists"


# Create filename for the matches indexes
matched_idx_filenames = [
    f"{dist_filename_prefix}_matched_idx_im{id_image+1}" for id_image in range(2)
]


# Define path to save the matches
matches_path = "computed_matches"
kp_filenames = [f"{dist_filename_prefix}_kp_{id_image}" for id_image in range(2)]
# Create filename for the pairs of keypoints
kp_pairs_filename = f"{dist_filename_prefix}_kp_pairs_arr"
# Create filename for the opencv Dmatches
matches_filename = f"{dist_filename_prefix}_matches"

# Create filename for the correct matches indexes
correct_matches_idxs_filename = f"{dist_filename_prefix}_correct_idxs"

# Create filename for 1 correct match
correct_match_filename_prefix = f"{dist_filename_prefix}_correct_match"

# Define path to save the chosen keypoints displayed
filtered_kp_path = "filtered_keypoints"

# Create filename for the log files containing everything that was printed
log_path = "execution_logs"
log_filename = f"{dist_filename_prefix}_log{sift_radical}{filt_radical}"

image_distances_filename = "image_distances"

# stack all folders paths
required_folders_paths = [
    descrip_path,
    dist_path,
    matches_path,
    filtered_kp_path,
    log_path,
]


# Create filename for cropped uint images (for SIFT)
cropped_ims_filenames = [
    f"{descrip_filename_prefixes[id_image]}_cropped_im" for id_image in range(2)
]

# Create filename for cropped float images (visualization)
cropped_float_ims_filenames = [
    f"{descrip_filename_prefixes[id_image]}_cropped_float_im" for id_image in range(2)
]
