import os
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import descriptor
import visu_descriptor

# add the path to the visualize_hessian folder
sys.path.append(os.path.join("..", "visualize_hessian"))
# import visualize_hessian.visu_hessian
import visu_hessian as vh

# return to the root directory
sys.path.append(os.path.join(".."))

# paths are relative to the directory where the terminal is opened

# load grayscale img
img = cv.imread(os.path.join("..", "images", "87_img_.png"), 0)
# plt.imshow(img, cmap="gray")
# plt.show()

# convert to float32
float32_img = vh.convert_uint8_to_float32(img)

# choose a keypoint
position = (200, 200)

# crop the image around the keypoint
radius = 50
sub_float32_img = float32_img[
    position[1] - radius : position[1] + radius + 1,
    position[0] - radius : position[0] + radius + 1,
]
sub_position = (radius, radius)


###########################
# Test compute_vector_histogram #
###########################

# eigvals, eigvects, gradients = vh.compute_hessian_gradient_subimage(
#     float32_img, border_size=1
# )
# gradients_norms = np.linalg.norm(gradients, axis=-1)
# orientations = descriptor.compute_orientations(float32_img, border_size=1)
# posdeg_orientations = descriptor.convert_angles_to_pos_degrees(orientations)
# histograms = descriptor.compute_vector_histogram(
#     gradients_norms, posdeg_orientations, position
# )
# print("histograms", histograms)
# display histogram figure
# descriptor.visu_descriptor.display_histogram(histograms[0, 0, :])
# descriptor.visu_descriptor.display_histogram(histograms[1, 1, :])
# descriptor.display_spatial_histograms(histograms)

##########################
# Test angle computation #
##########################

# # Define the matrix
# A = np.array([[1, 2], [3, 4]])

# # Compute the eigenvalues and eigenvectors
# eigenvalues, eigenvectors = np.linalg.eig(A)

# # The eigenvectors are the columns of the eigenvectors matrix. To return them as rows, transpose the matrix.
# eigenvectors = eigenvectors.T

# print(eigenvectors, eigenvectors.shape)


# # vects = np.full((1, 1, 2, 2), fill_value=1 / np.sqrt(2), dtype=np.float32)
# vects = np.array([[[1 / np.sqrt(2), 1 / np.sqrt(2)]], [[1, 1]]], dtype=np.float32)
# horiz_angles = descriptor.compute_horiz_angles(vects) * 180 / np.pi
# print("horiz_angles", horiz_angles)

# eigvals, eigvects, gradients = vh.compute_hessian_gradient_subimage(
#     float32_img[:50, :50], border_size=1
# )

# print(eigvects[:, :, 0, :].shape)  # y=1, x = 1

# # compute multiple angles at once
# vects1 = eigvects[:, :, 0, :]
# print(vects1)
# horiz_angles = descriptor.compute_horiz_angles(vects1) * 180 / np.pi
# print("horiz_angles", horiz_angles)

# # compute 1 single angle
# y, x = 1, 1
# v1 = eigvects[y, x, 0]
# print(v1)
# h_vect = np.array([0, 1], dtype=np.float32)
# h_ang1 = descriptor.compute_angle(v1, h_vect) * 180 / np.pi
# print("h_ang1", h_ang1)
# print("horiz_angles[y, x]", horiz_angles[y, x])


# # compute first principal directions of the Hessian matrix
# principal_directions1 = descriptor.compute_horiz_angles(eigvects[:, :, 0, :])

# # convert and rescale angles in [0, 360[
# posdeg_principal_directions1 = descriptor.convert_angles_to_pos_degrees(
#     principal_directions1
# )

# A = np.array([[1, 2], [1, 2]])
# eigvals, eigvects = np.linalg.eig(A)
# print("eigvals", eigvals)
# print("eigvects", eigvects)

a = 2


# #################################
# # Test compute_features_overall_posneg #
# #################################

# features_overall_posneg = descriptor.compute_features_overall_posneg(float32_img, border_size=1)

# # # print("features_overall_posneg", features_overall_posneg)

# # ######################################
# # # Test compute_descriptor_histograms_posneg #
# # ######################################

# descriptor_histos = descriptor.compute_descriptor_histograms_posneg(features_overall_posneg, position)

# # # print("histos", histos)

# # #################
# # # Visualization #
# # #################

# # # titles = ["positive eigenvalues", "negative eigenvalues", "gradients"]
# # # for id_value in range(len(histos)):
# # #     descriptor.display_spatial_histograms(histos[id_value], titles[id_value])

# values_names = ["positive eigenvalues", "negative eigenvalues", "gradients"]
# descriptor.display_descriptor(descriptor_histos, values_names=values_names)


# ################################
# Test compute_overall_features2 #
##################################

# features_overall_abs = descriptor.compute_features_overall_abs(float32_img, border_size=1)


#######################################
# Test compute_descriptor_histograms_1_2 #
#######################################

# descriptor_histos2 = descriptor.compute_descriptor_histograms_1_2(
#     features_overall_abs, position
# )


# # #################
# # # Visualization 2 #
# # #################

values_names2 = ["1st eigenvalues", "2nd eigenvalues", "gradients"]
# descriptor.display_descriptor(descriptor_histos2, values_names=values_names2)

#####################
# test rotate_pixel #
#####################

# arr = np.arange(25).reshape((5, 5)).astype(np.float32)
# kp_x, kp_y = 2, 2
# angle = 45.0
# arr_rotated = np.zeros_like(arr, dtype=np.float32)
# for i in range(arr.shape[0]):
#     for j in range(arr.shape[1]):
#         i_rot, j_rot = descriptor.rotate_pixel(i, j, kp_x, kp_y, angle)
#         arr_rotated[i_rot, j_rot] = arr[i, j]
# print("arr", arr)
# print("arr_rotated", arr_rotated)

import scipy.ndimage as ndimage


# a = 2


#######################################
# Test compute_descriptor_histograms_1_2_rotated #
#######################################
features_overall_abs = descriptor.compute_features_overall_abs(
    sub_float32_img, border_size=1
)

descriptor_histos2_rotated = descriptor.compute_descriptor_histograms_1_2_rotated(
    features_overall_abs, sub_position
)

values_names2 = ["1st eigenvalues", "2nd eigenvalues", "gradients"]
# visu_descriptor.display_descriptor(
#     descriptor_histos2_rotated, values_names=values_names2
# )


###################################
# Test display_matched_descriptor #
###################################

position2 = (200, 200)
sub_float32_img2 = float32_img[
    position2[1] - radius : position2[1] + radius + 1,
    position2[0] - radius : position2[0] + radius + 1,
]
features_overall_abs2 = descriptor.compute_features_overall_abs(
    sub_float32_img2, border_size=1
)
# global normalization
descriptor_histos2_rotated2_global = descriptor.compute_descriptor_histograms_1_2_rotated(
    features_overall_abs2, sub_position, normalization_mode="global"
)
visu_descriptor.display_matched_histograms(
    descriptor_histos2_rotated[0], descriptor_histos2_rotated2_global[0], hist_title="1st eigenvalues, global normalization"
)
# local normalization
descriptor_histos2_rotated2_local = descriptor.compute_descriptor_histograms_1_2_rotated(
    features_overall_abs2, sub_position, normalization_mode="local"
)
visu_descriptor.display_matched_histograms(
    descriptor_histos2_rotated[0], descriptor_histos2_rotated2_local[0], hist_title="1st eigenvalues, local normalization"
)
plt.show()
# visu_descriptor.display_matched_descriptors(
#     descriptor_histos2_rotated, descriptor_histos2_rotated2
# )


# #############################
# # Tests of unused functions #
# #############################


# ######################
# # Test create_1D_gaussian_kernel
# ######################

# kernel = descriptor.create_1D_gaussian_kernel(1.6)
# # print(kernel)
# # plt.plot(kernel)
# # plt.show()

# ######################
# # Test convolve_2D_gaussian
# ######################

# # convolution with a 2D separable Gaussian kernel
# convolved_img = descriptor.convolve_2D_gaussian(img, 1)
# # plt.imshow(convolved_img, cmap="gray")
# # plt.show()

# # convolve with a 2D Gaussian kernel
# convolved_img2 = cv.GaussianBlur(img, (0, 0), 1.6)
# # plt.imshow(convolved_img2, cmap="gray")
# # plt.show()

# ######################
# # Test compute_gaussian_mean
# ######################
# # 1D test
# # arr = np.ones((11, 11), dtype=np.float32)
# # print("shape", arr.shape)
# # g_mean = descriptor.compute_gaussian_mean(arr, 1.6)
# # print("g_mean", g_mean)
# # # 2D test
# # arr2 = np.ones((11, 11, 2), dtype=np.float32)
# # g_mean2 = descriptor.compute_gaussian_mean(arr2, 1.6)
# # print("g_mean2", g_mean2)
