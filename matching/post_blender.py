# Testing pipeline after Blender script execution
from matching import kp_filter
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from saving import load_matches, load_keypoints

#TODO: Filter matches, not keypoints

if __name__ == '__main__':
    im_folder = '../data/blender/blue_cylinder_more_light/'
    im_1 = cv.imread(im_folder + 'left.png', cv.IMREAD_GRAYSCALE)
    im_2 = cv.imread(im_folder + 'right.png', cv.IMREAD_GRAYSCALE)
    kp1_filename = im_folder + "keypoints_1.txt"
    kp2_filename = im_folder + "keypoints_2.txt"
    idx_filename = im_folder + "idxs.npy"
    matches_filename = im_folder + 'matches.txt'
    matches = load_matches(matches_filename)
    idxs = np.load(im_folder + 'idxs.npy')
    kp1 = load_keypoints(kp1_filename)
    kp2 = load_keypoints(kp2_filename)
    #kp1_filtered, kp2_filtered = kp_filter(kp1_filename, kp2_filename, idxs)
    good_matches = [matches[i] for i in idxs]

    matches_img = cv.drawMatchesKnn(
        # Warning : the number of matches to draw is not specified here
        img1=im_1,
        keypoints1=kp1,
        img2=im_2,
        keypoints2=kp2,
        matches1to2=good_matches,
        outImg=None,
        singlePointColor=(255, 0, 0)
    )

    plt.figure(figsize=(10, 5))
    plt.imshow(matches_img)
    plt.axis('off')
    plt.show()
