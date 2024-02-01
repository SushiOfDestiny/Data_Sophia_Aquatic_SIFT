# Entire testing pipeline before Blender script execution
import cv2 as cv
import numpy as np
from matching import get_keypoint_pairs, draw_good_keypoints
from saving import save_keypoints, save_matches, save_kp_pairs_to_arr
import sys
sys.path.append('../')
from visualise_hessian.visu_hessian import convert_uint8_to_float32
import matplotlib.pyplot as plt

if __name__ == '__main__':
    im_folder = '../data/blender/demo_cube/'
    im_name_1 = im_folder + 'left.png'
    im_name_2 = im_folder + 'right.png'
    kp_pairs, matches, kp_1, kp_2 = get_keypoint_pairs(im_name_1, im_name_2)
    im_1 = cv.imread(im_name_1, cv.IMREAD_GRAYSCALE)
    im_2 = cv.imread(im_name_2, cv.IMREAD_GRAYSCALE)
    save_keypoints(kp_1, im_folder + 'keypoints_1.txt')
    save_keypoints(kp_2, im_folder + 'keypoints_2.txt')
    save_matches(matches, im_folder + 'matches.txt')
    save_kp_pairs_to_arr(kp_pairs, im_folder + 'kp_pairs_arr')
    print('-------')
    print('Pre-blender processing finished')
