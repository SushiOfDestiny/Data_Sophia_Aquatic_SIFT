import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import visu_hessian as vh
import descriptor as desc
import visu_descriptor as visu_desc

from datetime import datetime

from numba import njit
import numba


img_folder = "../data/blender/rocks/"
im_name1 = "left"
im_name2 = "right"
im_names = [im_name1, im_name2]
im_ext = "png"

ims = [
    cv.imread(f"{img_folder}/{im_names[i]}.{im_ext}", cv.IMREAD_GRAYSCALE)
    for i in range(2)
]

croppings = [slice(0, 1080), slice(0, 1920), slice(0, 1080), slice(0, 1920)]
ims[0] = ims[0][croppings[0], croppings[1]]
ims[1] = ims[1][croppings[2], croppings[3]]

# compute float32 versions for calculations
float_ims = [vh.convert_uint8_to_float32(ims[i]) for i in range(2)]

# arbitrary sigma
blur_sigma = 1.0
float_ims = [desc.convolve_2D_gaussian(float_ims[i], blur_sigma) for i in range(2)]

# compute feature overall
before = datetime.now()
print(f"feat computation beginning:", before)

overall_features = desc.compute_features_overall_abs(float_ims[0])
after = datetime.now()
print(f"feat computation end", after)
print(" feat compute time", after - before)

# before = datetime.now()
# print(f"feat computation beginning:", before)
# overall_features = desc.compute_features_overall_abs(float_ims[0])
# after = datetime.now()
# print(f"feat computation end", after)
# print("compute time", after - before)

# compute descriptor at 1 position
# pos = [50, 50]
# before = datetime.now()
# print("1 desc computation beginning", before)

# descriptors = desc.compute_descriptor_histograms_1_2_rotated(
#     overall_features_1_2=overall_features, kp_position=pos
# )

# after = datetime.now()
# print("1 desc computation end")
# print(f"desc computation end", after)
# print("compute time", after - before)

# compute descriptors for some pixels
# y_start = 100
# y_length = 100
# x_start = 100
# x_length = 5

# before = datetime.now()
# print("desc computation beginning", before)
# for i in range(y_start, y_start + y_length):
#     for j in range(x_start, x_start + x_length):
#         desc.compute_descriptor_histograms_1_2_rotated(
#             overall_features_1_2=overall_features,
#             kp_position=(i, j))
# after = datetime.now()
# print("desc computation end")
# print("desc computation end", after)
# print("desc compute time", after - before)


@njit(parallel=True)
def compute_desc_pixels(overall_features, y_start, y_length, x_start, x_length):
    for i in numba.prange(y_start, y_start + y_length):
        for j in range(x_start, x_start + x_length):
            desc.compute_descriptor_histograms_1_2_rotated(
                overall_features_1_2=overall_features, kp_position=(i, j)
            )


y_start = 100
y_length = 100
x_start = 100
x_length = 100

before = datetime.now()
print("desc computation beginning", before)
compute_desc_pixels(overall_features, y_start, y_length, x_start, x_length)
after = datetime.now()
print("desc computation end")
print("desc computation end", after)
print("desc compute time", after - before)
