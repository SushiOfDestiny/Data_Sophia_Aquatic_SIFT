# present file stores all hyperparameters required to run the whole descriptor computation and blender matching pipeline

# define images to load
relative_path = "../data"
img_folder = "blender/rocks"
photo_name = "rock_1"
im_name1 = "left"
im_name2 = "right"
im_names = (im_name1, im_name2)
im_ext = "png"

# set the coordinates of the subimages

# 1st set
# y_starts = [386, 459]
# y_lengths = [100, 100]
# x_starts = [803, 806]
# x_lengths = [200, 200]

# 2nd set
y_starts = [530, 560]
y_lengths = [20, 20]
x_starts = [1200, 1200]
x_lengths = [20, 20]

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
