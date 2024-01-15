import os
import sys
import logging
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import visu_hessian

logger = logging.getLogger(__name__)
print("oui")

######################
# load grayscale image
######################

# img_path = "../images"
im_name = "dumbbell"
img_ext = "jpg"
# img = cv.imread(f"{img_path}/{im_name}.{img_ext}", 0)
img = cv.imread(f"{im_name}.{img_ext}", 0)
# # show image
# # plt.imshow(img, cmap="gray")
# # plt.show()

# define folder to save images
img_folder = "zoomed_kp"
img_resolution = 400  # in dpi

#########################################
# calculate sift keypoints and descriptors
#########################################

sift = cv.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)

# draw keypoints on image
img_kp = cv.drawKeypoints(
    img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
nb_kp = len(keypoints)
# plt.imshow(img_kp)
# plt.title(f"{nb_kp} SIFT keypoints")
# # # save figure
# # plt.savefig(
# #     f"{im_name}_sift.png",
# #     dpi=img_resolution,
# # )
# plt.show()

# draw colormap of eigenvalues of Hessian matrix for 1 keypoint
kp0 = keypoints[200]
y_kp0, x_kp0 = np.round(kp0.pt).astype(int)
position = (y_kp0, x_kp0)
# print(kp0.pt)

################################
# test convert_uint8_to_float32
################################

img_float32 = visu_hessian.convert_uint8_to_float32(img)
print(img_float32.min(), img_float32.max())
# plt.imshow(img_float32, cmap="gray")
# plt.show()


##################################
# test crop_subimage_around_keypoint
##################################

sub_img = visu_hessian.crop_image_around_keypoint(img_float32, position, 30)
print(sub_img.min(), sub_img.max())

# plt.imshow(sub_img, cmap="gray")
# plt.show()

##################################
# test compute_gradient_subimage
##################################

eigvals, eigvects, gradients = visu_hessian.compute_hessian_gradient_subimage(sub_img)
print(eigvals.shape, eigvects.shape, gradients.shape)


###################################
# Test visualize_curvature_values #
###################################
y_kp0, x_kp0 = np.round(kp0.pt).astype(int)

# eigval_fig = visu_hessian.visualize_curvature_values(img_float32, kp0, 30)

# plt.figure(eigval_fig.number)
# plt.show()

# # save figure
# eigval_fig.savefig(
#     f"zoomed_kp/zoomed_{im_name}_kp_{y_kp}_{x_kp}_{zoom_radius}.png",
#     dpi="figure",
# )

# draw colormap of eigenvalues of Hessian matrix for some keypoints
# zoom_radius = 10
# for kp in keypoints[50:301:50]:
#     visualize_curvature_values(im_name, img, kp, zoom_radius)

#######################################
# # Test visualize_curvature_directions #
#######################################

# zoom_radius = 30
# eig_fig = visu_hessian.visualize_curvature_directions(img_float32, kp0, zoom_radius)

# plt.figure(eig_fig.number)
# plt.show()

# # save figure
# eig_fig.savefig(
#     f"{img_folder}/zoomed_{im_name}_kp_{y_kp0}_{x_kp0}_{zoom_radius}_eigvects.png",
#     dpi=img_resolution,
# )

# # test a bunch of keypoints
# zoom_radius = 30
# start_kp = 50
# nb_kp = 5
# step_idx = 50
# end_kp = start_kp + nb_kp * step_idx
# for kp in keypoints[start_kp:end_kp:step_idx]:
#     eigvec_fig = visualize_curvature_directions(img, kp, zoom_radius)
#     plt.figure(eigvec_fig.number)
#     plt.show()

# ##############
# # test visualise_curvature_directions_ax_sm
# ##############

# # create figure and ax
# fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# # compute eigenvectors and add them to the ax
# sm = visu_hessian.visualize_curvature_directions_ax_sm(
#     img_float32, kp0, zoom_radius=15, ax=ax
# )

# # add the colorbar of the colormap of the arrows
# fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

# # add legend
# fig.suptitle(f"SIFT Keypoint {y_kp0}, {x_kp0} (in red) from 2nd function", fontsize=10)

# plt.show()

###############
# test downsample_array
###############

arr = np.ones((10, 10, 2))
d_arr = visu_hessian.downsample_array(arr, 2)
print(d_arr[:, :, 0])
print(d_arr[:, :, 1])

##############
# test


##############
# test visualise_curvature_directions_ax_sm_unifinished
##############

# create figure and ax
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# compute eigenvectors and add them to the ax
sm = visu_hessian.visualize_curvature_directions_ax_sm_unfinished(
    img_float32, kp0, zoom_radius=15, ax=ax
)

# add the colorbar of the colormap of the arrows
fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

# add legend
fig.suptitle(f"unfinished", fontsize=10)

plt.show()


################
# Test gradients
################

# grad_fig = visu_hessian.visualize_gradients(img_float32, kp0, zoom_radius)

# plt.figure(grad_fig.number)
# plt.show()


##################
# test visualize_gradients_ax_sm
##################

# # create figure and ax
# fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# # compute eigenvectors and add them to the ax
# sm = visu_hessian.visualize_gradients_ax_sm(img_float32, kp0, zoom_radius=15, ax=ax)

# # add the colorbar of the colormap of the arrows
# fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

# # add legend
# fig.suptitle(f"SIFT Keypoint {y_kp0}, {x_kp0} (in red) from 2nd function", fontsize=10)

# plt.show()

##################
# test visualize_gradients_ax_sm_unfinished
##################

# create figure and ax
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# compute eigenvectors and add them to the ax
sm = visu_hessian.visualize_gradients_ax_sm_unfinished(
    img_float32, kp0, zoom_radius=15, ax=ax
)

# add the colorbar of the colormap of the arrows
fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

# add legend
fig.suptitle(f"unfinished", fontsize=10)

plt.show()

##################
# Test compare directions
##################

# # load grayscale image
# img_path = "../data"
# im_name1 = "cube-0"
# im_name2 = "cube-1"
# img_ext = "png"
# g_img1 = cv.imread(f"{img_path}/{im_name1}.{img_ext}", 0)
# g_img2 = cv.imread(f"{img_path}/{im_name2}.{img_ext}", 0)

# # compute float32 images
# float32_g_img1 = visu_hessian.convert_uint8_to_float32(g_img1)
# float32_g_img2 = visu_hessian.convert_uint8_to_float32(g_img2)

# # calculate sift keypoints and descriptors
# sift = cv.SIFT_create()
# keypoints1, descriptors1 = sift.detectAndCompute(g_img1, None)
# keypoints2, descriptors2 = sift.detectAndCompute(g_img2, None)

# # match descriptors
# bf = cv.BFMatcher()
# matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# # Get keypoints of first match
# kp1 = keypoints1[matches[0][0].queryIdx]
# kp2 = keypoints2[matches[0][0].trainIdx]

# # compare directions
# dir_fig = visu_hessian.compare_directions(
#     float32_g_img1, float32_g_img2, kp1, kp2, zoom_radius=15
# )

# # show figure
# plt.figure(dir_fig.number)
# plt.show()

# ##################
# # Test compare gradients
# ##################

# grad_fig = visu_hessian.compare_gradients(
#     float32_g_img1, float32_g_img2, kp1, kp2, zoom_radius=15
# )

# # show figure
# plt.figure(grad_fig.number)
# plt.show()
