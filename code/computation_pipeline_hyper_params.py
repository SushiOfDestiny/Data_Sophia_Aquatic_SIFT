# # present file stores all hyperparameters required to run the whole descriptor computation and blender matching pipeline

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

# # set the coordinates of the subimages

# y_starts = [386, 459]
# y_lengths = [100, 100]
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
# # Pipeline 2 #
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


##############
# Pipeline 4 #
##############

# define images to load
relative_path = "../data"
# img_folder = "blender/rocks"
img_folder = "irl/scene_zipped"
photo_name = "irl_rock_1"
im_name1 = "scene_l_1_u"
im_name2 = "scene_r_1_u"
im_names = (im_name1, im_name2)
im_ext = "jpeg"

# set the coordinates of the subimages

y_starts = [910, 1140]
y_lengths = [100, 100]
x_starts = [1670, 1750]
x_lengths = [100, 100]

# define sigma for the gaussian blur, put 0.0 to not blur
blur_sigma = 0.0

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
