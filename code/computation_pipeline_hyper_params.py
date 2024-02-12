# present file stores all hyperparameters required to run the whole descriptor computation and blender matching pipeline

# ##############
# # Pipeline 1 #
# ##############
# # define images to load
# relative_path = "../data"
# # img_folder = "blender/rocks"
# img_folder = "blender/rocks"
# photo_name = "rock_1"
# im_name1 = "left"
# im_name2 = "right"
# im_names = (im_name1, im_name2)
# im_ext = "png"
# photo_name = "rock_1"
# im_name1 = "left"
# im_name2 = "right"
# im_names = (im_name1, im_name2)
# im_ext = "png"

# # choose if sift must be used
# use_sift = True
# # choose if sift must be used
# use_sift = True

# # set the coordinates of the subimages
# # set the coordinates of the subimages

# y_starts = [386, 459]
# y_lengths = [200, 200]
# x_starts = [803, 806]
# x_lengths = [200, 200]

# # define sigma for the gaussian blur
# blur_sigma = 1.0

# # define the border size
# border_size = 1

# # define descriptor parameters
# nb_bins = 3
# bin_radius = 2
# delta_angle = 5.0
# sigma = 0
# normalization_mode = "global"
# neigh_radius = (2 * bin_radius + 1) * nb_bins // 2 + bin_radius
# nb_angular_bins = int(360.0 / delta_angle) + 1

# # set the distance type, between "min" and "all"
# distance_type = "min"

# # define the blender filtering precision threshold
# epsilon = None


# ##############
# # Pipeline 2 # Related to SIFT on whole image
# ##############

# # define images to load
# relative_path = "../data"
# # img_folder = "blender/rocks"
# img_folder = "blender/rocks"
# photo_name = "rock_1"
# im_name1 = "left"
# im_name2 = "right"
# im_names = (im_name1, im_name2)
# im_ext = "png"

# # choose if sift must be used
# use_sift = True

# # set the coordinates of the subimages
# y_starts = [530, 560]
# y_lengths = [20, 20]
# x_starts = [1200, 1200]
# x_lengths = [20, 20]

# # define sigma for the gaussian blur
# blur_sigma = 1.0

# # define the border size
# border_size = 1

# # define descriptor parameters
# nb_bins = 3
# bin_radius = 2
# delta_angle = 5.0
# sigma = 0
# normalization_mode = "global"
# neigh_radius = (2 * bin_radius + 1) * nb_bins // 2 + bin_radius
# nb_angular_bins = int(360.0 / delta_angle) + 1

# # set the distance type, between "min" and "all"
# distance_type = "min"

# # define the blender filtering precision threshold
# epsilon = None


# # #############
# # Pipeline 2.2 # for sift subimage
# # #############

# # define images to load
# relative_path = "../data"
# # img_folder = "blender/rocks"
# img_folder = "blender/rocks"
# photo_name = "rock_1"
# im_name1 = "left"
# im_name2 = "right"
# im_names = (im_name1, im_name2)
# im_ext = "png"

# # choose if sift must be used
# use_sift = True

# # set the coordinates of the subimages
# y_starts = [530, 560]
# y_lengths = [200, 200]
# x_starts = [1200, 1200]
# x_lengths = [200, 200]

# # define sigma for the gaussian blur
# blur_sigma = 1.0

# # define the border size
# border_size = 1

# # define descriptor parameters
# nb_bins = 3
# bin_radius = 2
# delta_angle = 5.0
# sigma = 0
# normalization_mode = "global"
# neigh_radius = (2 * bin_radius + 1) * nb_bins // 2 + bin_radius
# nb_angular_bins = int(360.0 / delta_angle) + 1

# # set the distance type, between "min" and "all"
# distance_type = "min"

# # define the blender filtering precision threshold
# epsilon = None


# ##############
# # Pipeline 3 #
# ##############

# # define images to load
# relative_path = "../data"
# # img_folder = "blender/rocks"
# img_folder = "blender/rock_1_rotated"
# photo_name = "rock_1_rot"
# im_name1 = "left"
# im_name2 = "right"
# im_names = (im_name1, im_name2)
# im_ext = "png"

# # choose if sift must be used
# use_sift = False

# # set the coordinates of the subimages

# y_starts = [510, 560]
# y_lengths = [50, 50]
# x_starts = [780, 775]
# x_lengths = [60, 60]

# # define sigma for the gaussian blur
# blur_sigma = 1.0

# # define the border size
# border_size = 1

# # define descriptor parameters
# nb_bins = 3
# bin_radius = 2
# delta_angle = 5.0
# sigma = 0
# normalization_mode = "global"
# neigh_radius = (2 * bin_radius + 1) * nb_bins // 2 + bin_radius
# nb_angular_bins = int(360.0 / delta_angle) + 1

# # set the distance type, between "min" and "all"
# distance_type = "min"

# # define the blender filtering precision threshold
# epsilon = None


# ##############
# # Pipeline 4 #
# ##############

# # define images to load
# relative_path = "../data"
# # img_folder = "blender/rocks"
# img_folder = "irl/scene_zipped"
# photo_name = "irl_rock_1"
# im_name1 = "scene_l_1_u"
# im_name2 = "scene_r_1_u"
# im_names = (im_name1, im_name2)
# im_ext = "jpeg"

# # choose if sift must be used
# use_sift = False

# # set the coordinates of the subimages

# y_starts = [910, 1140]
# y_lengths = [100, 100]
# x_starts = [1670, 1750]
# x_lengths = [100, 100]

# # define sigma for the gaussian blur, put 0.0 to not blur
# blur_sigma = 0.0

# # define the border size
# border_size = 1
# # define the border size
# border_size = 1

# # define descriptor parameters
# nb_bins = 3
# bin_radius = 2
# delta_angle = 5.0
# sigma = 0
# normalization_mode = "global"
# neigh_radius = (2 * bin_radius + 1) * nb_bins // 2 + bin_radius
# nb_angular_bins = int(360.0 / delta_angle) + 1
# # define descriptor parameters
# nb_bins = 3
# bin_radius = 2
# delta_angle = 5.0
# sigma = 0
# normalization_mode = "global"
# neigh_radius = (2 * bin_radius + 1) * nb_bins // 2 + bin_radius
# nb_angular_bins = int(360.0 / delta_angle) + 1

# # set the distance type, between "min" and "all"
# distance_type = "min"
# # set the distance type, between "min" and "all"
# distance_type = "min"

# # define the blender filtering precision threshold
# epsilon = None
# # define the blender filtering precision threshold
# epsilon = None


# ##############
# # Pipeline 5 #
# ##############

# # define images to load
# relative_path = "../data"
# # img_folder = "blender/rocks"
# img_folder = "blender/rocks"
# photo_name = "rock_1"
# im_name1 = "left"
# im_name2 = "right"
# im_names = (im_name1, im_name2)
# im_ext = "png"

# # choose if sift must be used
# use_sift = True

# # set the coordinates of the subimages
# y_starts = [10, 10]
# y_lengths = [1060, 1060]
# x_starts = [10, 10]
# x_lengths = [1900, 1900]

# # define sigma for the gaussian blur
# blur_sigma = 1.0

# # define the border size
# border_size = 1

# # define descriptor parameters
# nb_bins = 3
# bin_radius = 2
# delta_angle = 5.0
# sigma = 0
# normalization_mode = "global"
# neigh_radius = (2 * bin_radius + 1) * nb_bins // 2 + bin_radius
# nb_angular_bins = int(360.0 / delta_angle) + 1

# # set the distance type, between "min" and "all"
# distance_type = "min"

# # define the blender filtering precision threshold
# epsilon = None


# ##############
# # Pipeline 4 #
# ##############

# # define images to load
# relative_path = "../data"
# # img_folder = "blender/rocks"
# img_folder = "irl/scene_zipped"
# photo_name = "irl_rock_1"
# im_name1 = "scene_l_1_u"
# im_name2 = "scene_r_1_u"
# im_names = (im_name1, im_name2)
# im_ext = "jpeg"

# # choose if sift must be used
# use_sift = True

# # set the coordinates of the subimages

# y_starts = [910, 1140]
# y_lengths = [100, 100]
# x_starts = [1670, 1750]
# x_lengths = [100, 100]

# # define sigma for the gaussian blur, put 0.0 to not blur
# blur_sigma = 0.0

# # define the border size
# border_size = 1

# # define descriptor parameters
# nb_bins = 3
# bin_radius = 2
# delta_angle = 5.0
# sigma = 0
# normalization_mode = "global"
# neigh_radius = (2 * bin_radius + 1) * nb_bins // 2 + bin_radius
# nb_angular_bins = int(360.0 / delta_angle) + 1

# # set the distance type, between "min" and "all"
# distance_type = "min"

# # define the blender filtering precision threshold
# epsilon = None


# #############
# # Pipeline 8 #
# ##############

# # define images to load
# relative_path = "../data"
# # img_folder = "blender/rocks"
# img_folder = "blender/rocks"
# photo_name = "rock_1"
# im_name1 = "left"
# im_name2 = "right"
# im_names = (im_name1, im_name2)
# im_ext = "png"

# # choose if sift must be used
# use_sift = False

# # set the coordinates of the subimages

# y_starts = [386, 459]
# y_lengths = [60, 60]
# x_starts = [803, 806]
# x_lengths = [60, 60]

# # define sigma for the gaussian blur
# blur_sigma = 1.0

# # define the border size
# border_size = 1

# # define descriptor parameters
# nb_bins = 3
# bin_radius = 2
# delta_angle = 5.0
# sigma = 0
# normalization_mode = "global"
# neigh_radius = (2 * bin_radius + 1) * nb_bins // 2 + bin_radius
# nb_angular_bins = int(360.0 / delta_angle) + 1

# # set the distance type, between "min" and "all"
# distance_type = "min"

# # define the blender filtering precision threshold
# epsilon = None


# ##############
# # Pipeline 9 # Test SIFT on cropped image
# ##############
# ##############
# # Pipeline 9 # Test SIFT on cropped image
# ##############

# # define images to load
# relative_path = "../data"
# # img_folder = "blender/rocks"
# img_folder = "blender/rocks"
# photo_name = "rock_1"
# im_name1 = "left"
# im_name2 = "right"
# im_names = (im_name1, im_name2)
# im_ext = "png"

# # choose if sift must be used instead of our home made descriptor
# use_sift = False

# # set the coordinates of the subimages
# y_starts = [386, 459]
# y_lengths = [200, 200]
# x_starts = [803, 806]
# x_lengths = [200, 200]

# # define sigma for the gaussian blur
# blur_sigma = 1.0

# # define the border size
# border_size = 1

# # define descriptor parameters
# nb_bins = 3
# bin_radius = 2
# delta_angle = 5.0
# sigma = 0
# normalization_mode = "global"
# neigh_radius = (2 * bin_radius + 1) * nb_bins // 2 + bin_radius
# nb_angular_bins = int(360.0 / delta_angle) + 1

# # set the distance type, between "min" and "all"
# distance_type = "min"

# # define the blender filtering precision threshold
# epsilon = None


##############
# Pipeline 10 # Test curve filter
##############

# define images to load
relative_path = "../data"
# img_folder = "blender/rocks"
img_folder = "blender/rocks"
photo_name = "rock_1"
im_name1 = "left"
im_name2 = "right"
im_names = (im_name1, im_name2)
im_ext = "png"

# choose if sift must be used instead of our home made descriptor
use_sift = False

# choose if filter
use_filt = False


# set the coordinates of the subimages
y_starts = [386, 459]
y_lengths = [100, 100]
x_starts = [803, 806]
x_lengths = [100, 100]

# define sigma for the gaussian blur
blur_sigma = 1.0

# define the border size
border_size = 1

# define descriptor parameters
nb_bins = 3
bin_radius = 2
delta_angle = 5.0
sigma = 0
normalization_mode = "global"
neigh_radius = (2 * bin_radius + 1) * nb_bins // 2 + bin_radius
nb_angular_bins = int(360.0 / delta_angle) + 1

# set the distance type, between "min" and "all"
distance_type = "min"

# define the blender filtering precision threshold
epsilon = None
blender_filename = "rocks"

use_sift = True
use_filt = False

# SIFT parameters
method_post = "lowe"
contrastThreshold = 0.0
edgeThreshold = 0.0
SIFTsigma = 1.6
distanceThreshold = 1e9

# ##############
# # Pipeline ? #
# ##############

# # define images to load
# relative_path = "../data"
# # define images to load
# relative_path = "../data"
# # img_folder = "blender/rocks"
# img_folder = "blender/rocks_2"
# photo_name = "rocks_2_13_deg_cam_to_scale"
# im_name1 = "rocks_2_13_deg_cam_to_scale_left"
# im_name2 = "rocks_2_13_deg_cam_to_scale_right"
# blender_filename = photo_name
# im_names = (im_name1, im_name2)
# im_ext = "png"

# # # choose if sift must be used
# # use_sift = False

# # set the coordinates of the subimages

# y_starts = [500, 500]
# y_lengths = [200, 200]
# x_starts = [840, 750]
# x_lengths = [200, 200]

# # define sigma for the gaussian blur
# blur_sigma = 1.0

# # define the border size
# border_size = 1

# # define descriptor parameters
# nb_bins = 3
# bin_radius = 2
# delta_angle = 5.0
# sigma = 0
# normalization_mode = "global"
# neigh_radius = (2 * bin_radius + 1) * nb_bins // 2 + bin_radius
# nb_angular_bins = int(360.0 / delta_angle) + 1

# # set the distance type, between "min" and "all"
# distance_type = "min"
# # define the blender filtering precision threshold
# epsilon = None
