import os
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import descriptor

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

######################
# Test create_1D_gaussian_kernel
######################

kernel = descriptor.create_1D_gaussian_kernel(1.6)
# print(kernel)
# plt.plot(kernel)
# plt.show()

######################
# Test convolve_2D_gaussian
######################

# convolution with a 2D separable Gaussian kernel
convolved_img = descriptor.convolve_2D_gaussian(img, 1)
# plt.imshow(convolved_img, cmap="gray")
# plt.show()

# convolve with a 2D Gaussian kernel
convolved_img2 = cv.GaussianBlur(img, (0, 0), 1.6)
# plt.imshow(convolved_img2, cmap="gray")
# plt.show()

######################
# Test compute_gaussian_mean
######################
# 1D test
# arr = np.ones((11, 11), dtype=np.float32)
# print("shape", arr.shape)
# g_mean = descriptor.compute_gaussian_mean(arr, 1.6)
# print("g_mean", g_mean)
# # 2D test
# arr2 = np.ones((11, 11, 2), dtype=np.float32)
# g_mean2 = descriptor.compute_gaussian_mean(arr2, 1.6)
# print("g_mean2", g_mean2)

#############################
#  Test compute_kp_features #
#############################

# compute global features
# eigvals, eigvects, gradients = vh.compute_hessian_gradient_subimage(
#     float32_img, border_size=1
# )
# orientations = descriptor.compute_orientations(float32_img, border_size=1)

# separate eigenvalues and eigenvectors
# features = [
#     eigvals[:, :, 0],
#     eigvals[:, :, 1],
#     eigvects[:, :, 0],
#     eigvects[:, :, 1],
#     gradients,
#     orientations,
# ]

# choose a keypoint
position = (100, 100)

# compact_features = descriptor.compute_compact_features_vect(features, position)
# print("compact_features", compact_features)

##############################
#  Test compact2flat_features_vect #
##############################

# flat_features = descriptor.compact2flat_features_vect(compact_features)
# print("flat_features", flat_features)

###########################
# Test compute_vector_histogram #
###########################
eigvals, eigvects, gradients = vh.compute_hessian_gradient_subimage(
    float32_img, border_size=1
)
values = np.linalg.norm(gradients, axis=-1)
orientations = descriptor.compute_orientations(float32_img, border_size=1)
histograms = descriptor.compute_vector_histogram(values, orientations, position)
# print("histograms", histograms)
# display histogram figure
descriptor.display_histogram(histograms[0, 0, :])


#################################
# Test compute_features_overall #
#################################

# features_overall = descriptor.compute_features_overall(float32_img, border_size=1)
# print("features_overall", features_overall)

###########################
# Test compute_descriptor #
###########################

# histos = descriptor.compute_descriptor(features_overall, position)
# print("histos", histos)
