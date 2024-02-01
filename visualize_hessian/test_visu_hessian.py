import os
import sys
import logging
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import visu_hessian as vh

# add the path to the descriptor folder
sys.path.append(os.path.join("..", "descriptor"))
# import visualize_hessian.visu_hessian
import descriptor as desc

# return to the root directory
sys.path.append(os.path.join(".."))

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
# img_folder = "zoomed_kp"
# img_resolution = 400  # in dpi

#########################################
# calculate sift keypoints and descriptors
#########################################

# sift = cv.SIFT_create()
# keypoints, descriptors = sift.detectAndCompute(img, None)

# # draw keypoints on image
# img_kp = cv.drawKeypoints(
#     img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )
# nb_kp = len(keypoints)
# plt.imshow(img_kp)
# plt.title(f"{nb_kp} SIFT keypoints")
# # # save figure
# # plt.savefig(
# #     f"{im_name}_sift.png",
# #     dpi=img_resolution,
# # )
# plt.show()

# draw colormap of eigenvalues of Hessian matrix for 1 keypoint
# kp0 = keypoints[90]
# x_kp0, y_kp0 = np.round(kp0.pt).astype(int)
# position = (x_kp0, y_kp0)
# print(kp0.pt)

################################
# test convert_uint8_to_float32
################################

img_float32 = vh.convert_uint8_to_float32(img)
# print(img_float32.min(), img_float32.max())
# plt.imshow(img_float32, cmap="gray")
# plt.show()


##################################
# test crop_subimage_around_keypoint
##################################
# zoom_radius = 30

# sub_img = vh.crop_image_around_keypoint(img_float32, position, zoom_radius)
# print(sub_img.min(), sub_img.max())

# plt.imshow(sub_img, cmap="gray")
# plt.show()

##################################
# test compute_gradient_subimage
##################################

# eigvals, eigvects, gradients = vh.compute_hessian_gradient_subimage(sub_img)
# print(eigvals.shape, eigvects.shape, gradients.shape)


###################################
# Test visualize_curvature_values #
###################################
# x_kp0, y_kp0 = np.round(kp0.pt).astype(int)

# eigval_fig = vh.visualize_curvature_values(img_float32, kp0, 30)

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
# eig_fig = vh.visualize_curvature_directions(img_float32, kp0, zoom_radius)

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

###############
# test downsample_array
###############

# arr = np.ones((10, 10, 2))
# d_arr = vh.downsample_array(arr, 2)
# print(d_arr[:, :, 0])
# print(d_arr[:, :, 1])

##############
# test


##############
# test visualise_curvature_directions_ax_sm
##############

# # create figure and ax
# fig, ax = plt.subplots(1, 1, figsize=(20, 20))

# # compute eigenvectors and add them to the ax
# sm = vh.visualize_curvature_directions_ax_sm(
#     img_float32, kp0, zoom_radius, ax=ax
# )

# # add the colorbar of the colormap of the arrows
# fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

# # add legend
# fig.suptitle(f"unfinished", fontsize=10)

# plt.show()


################
# Test gradients
################

# grad_fig = vh.visualize_gradients(img_float32, kp0, zoom_radius)

# plt.figure(grad_fig.number)
# plt.show()

##################
# test visualize_gradients_ax_sm
##################

# # create figure and ax
# fig, ax = plt.subplots(1, 1, figsize=(20, 20))

# # compute eigenvectors and add them to the ax
# sm = vh.visualize_gradients_ax_sm(img_float32, kp0, zoom_radius, ax=ax)

# # add the colorbar of the colormap of the arrows
# fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

# # add legend
# fig.suptitle(f"unfinished", fontsize=10)

# plt.show()

##################
# Test compare directions
##################

# load grayscale image
# img_path = "../data"
# im_name1 = "cube-0"
# im_name2 = "cube-1"
# img_ext = "png"
# g_img1 = cv.imread(f"{img_path}/{im_name1}.{img_ext}", 0)
# g_img2 = cv.imread(f"{img_path}/{im_name2}.{img_ext}", 0)

# # compute float32 images
# float32_g_img1 = vh.convert_uint8_to_float32(g_img1)
# float32_g_img2 = vh.convert_uint8_to_float32(g_img2)

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

# compare directions
# dir_fig = vh.compare_directions(
#     float32_g_img1, float32_g_img2, kp1, kp2, zoom_radius=30
# )

# # show figure
# plt.figure(dir_fig.number)
# plt.show()

# ##################
# # Test compare gradients
# ##################

# grad_fig = vh.compare_gradients(
#     float32_g_img1, float32_g_img2, kp1, kp2, zoom_radius=15
# )

# # show figure
# plt.figure(grad_fig.number)
# plt.show()

########################
# Test rotate_subimage #
########################

# position = (500, 500)
# zoom_radius = 150
# orientation = 45.0
# bigger_radius = 2 * int(0.5 * np.ceil(zoom_radius * np.sqrt(2))) + 1

# big_sub_image = vh.crop_image_around_keypoint(img_float32, position, bigger_radius)
# rotated_big_sub_image = vh.rotate_subimage(
#     img_float32, position[0], position[1], orientation, zoom_radius
# )
# fig, ax = plt.subplots(1, 2, figsize=(40, 20))
# ax[0].imshow(big_sub_image, cmap="gray")
# ax[1].imshow(rotated_big_sub_image, cmap="gray")
# plt.show()

######################################
# visualize_curvature_values_rotated #
######################################
img_path = "../data"
im_name = "dumbbell"
img_ext = "jpg"
# img = cv.imread(f"{img_path}/{im_name}.{img_ext}", 0)
img = cv.imread(f"{im_name}.{img_ext}", 0)

position = (500, 500)
kp = cv.KeyPoint(x=position[0], y=position[1], size=1)
zoom_radius = 30

kp_gradient_orientation = desc.compute_orientation(img_float32, position)
kp_gradient_orientation = desc.convert_angles_to_pos_degrees(kp_gradient_orientation)
angle_2 = kp_gradient_orientation + 45
print("kp_gradient_orientation", kp_gradient_orientation)
figs = [
    vh.visualize_curvature_values_rotated(img_float32, kp, angle, zoom_radius)
    for angle in (kp_gradient_orientation, angle_2)
]

for fig in figs:
    plt.figure(fig.number)
    plt.show()


######################################
# visualize_gradients_ax_sm_rotated #
######################################
# position = (500, 500)
# kp = cv.KeyPoint(x=position[0], y=position[1], size=1)
# zoom_radius = 30

# kp_gradient_orientation = desc.compute_orientation(img_float32, position)
# kp_gradient_orientation = desc.convert_angles_to_pos_degrees(kp_gradient_orientation)
# angle_2 = kp_gradient_orientation + 45
# print("kp_gradient_orientation", kp_gradient_orientation)

# figs = [
#     vh.visualize_curvature_values_rotated(img_float32, kp, angle, zoom_radius)
#     for angle in (kp_gradient_orientation, angle_2)
# ]

# for fig in figs:
#     plt.figure(fig.number)
#     plt.show()
