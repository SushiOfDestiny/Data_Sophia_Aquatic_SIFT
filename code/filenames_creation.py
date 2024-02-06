import numpy as np

from computation_pipeline_hyper_params import (
    relative_path,
    img_folder,
    photo_name,
    im_names,
    im_ext,
    blur_sigma,
    y_starts,
    y_lengths,
    x_starts,
    x_lengths,
    distance_type,
)

# Goal: create the filenames for all objects save during the computation pipeline

# Create filename where to load the original images
original_imgs_path_prefix = f"{relative_path}/{img_folder}"

# Create the filename for the blurred images
blurred_imgs_path = f"{original_imgs_path_prefix}_blurred_ims_sig{int(blur_sigma)}"

# Define path to save the descriptors and coordinates
descrip_path = "computed_descriptors"
# Create filename for the descriptors and coordinates
descrip_filename_prefixes = [
    f"{photo_name}_{im_names[id_image]}_y_{y_starts[id_image]}_{y_lengths[id_image]}_x_{x_starts[id_image]}_{x_lengths[id_image]}"
    for id_image in range(2)
]
descrip_filenames = [
    f"{descrip_filename_prefixes[id_image]}_descs.npy" for id_image in range(2)
]

# Define suffix for filename
dist_type_suffix = "" if distance_type == "all" else "_min"

# Define path to save the distances and matches
dist_path = "computed_distances"
# Create filename for the distances and matches
dist_filename_prefix = f"{photo_name}_y_{y_starts[0]}_{y_starts[1]}_{y_lengths[0]}_{y_lengths[1]}_x_{x_starts[0]}_{x_starts[1]}_{x_lengths[0]}_{x_lengths[1]}{dist_type_suffix}"

# Create filename for the matches indexes
matched_idx_filename_prefix = 
matched_idx_filenames = [
    f"{dist_filename_prefix}_matched_idx_im{id_image+1}.npy" for id_image in range(2)
]
# Create filename for the keypoints coordinates
kp_coords_filename_prefixes = descrip_filename_prefixes
kp_coords_filenames = [
    f"{kp_coords_filename_prefixes[id_image]}_coords.npy" for id_image in range(2)
]


# Define path to save the matches
matches_path = "computed_matches"
kp_filenames = [f"{dist_filename_prefix}_kp_{id_image}.txt" for id_image in range(2)]
# Create filename for the pairs of keypoints
kp_pairs_filename = f"{dist_filename_prefix}_kp_pairs_arr"
# Create filename for the opencv Dmatches
matches_filename = f"{dist_filename_prefix}_matches.txt"

# Create filename for the correct matches indexes
correct_matches_idxs_filename = f"{dist_filename_prefix}_correct_idxs"
                    
# Create filename for 1 correct match
correct_match_filename_prefix = f"{dist_filename_prefix}_correct_match"