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
# y_starts = [386, 459]
# y_lengths = [10, 10]
# x_starts = [803, 806]
# x_lengths = [20, 20]

y_starts = [10, 10]
y_lengths = [10, 10]
x_starts = [20, 20]
x_lengths = [20, 20]

# define sigma for the gaussian blur
blur_sigma = 1.0

border_size = 1

# set the distance type, between "min" and "all"
distance_type = "min"

# define the blender filtering precision threshold
epsilon = None
