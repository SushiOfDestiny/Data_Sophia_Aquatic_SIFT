import cv2 as cv
import matplotlib.pyplot as plt
import sys
import numpy as np
from saving import load_keypoints, save_kp_pairs_to_arr
sys.path.append('../')
from visualise_hessian.visu_hessian import convert_uint8_to_float32

def get_keypoint_pairs(im_name1, im_name2, method_post='lowe'):
    '''Default post matching method = Lowe's ratio test

    Matching is done through a knn match

    WARNING : CrossCheck is experimental, do not use
    
    Returns :
    - List of OpenCV keypoints
    - List of OpenCV matches'''

    # Get images
    im1 = cv.imread(im_name1, cv.IMREAD_GRAYSCALE)
    im2 = cv.imread(im_name2, cv.IMREAD_GRAYSCALE)

    cv.imshow('', im1)
    cv.waitKey()
    cv.imshow('', im2)
    cv.waitKey()

    # Compute float32 versions for calculations
    float_im1 = convert_uint8_to_float32(im1)
    float_im2 = convert_uint8_to_float32(im2)

    # Smooth images
    float_im1 = cv.GaussianBlur(float_im1, ksize=[0, 0], sigmaX=1, sigmaY=0)
    float_im2 = cv.GaussianBlur(float_im2, ksize=[0, 0], sigmaX=1, sigmaY=0)


    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    # BFMatcher with default params
    # default norm is L2
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []

    if method_post == 'crossCheck':
        # Alternative: crossCheck=True
        bf_CC = cv.BFMatcher(crossCheck=True)
        matches_CC = bf_CC.match(des1, des2)
        good_sorted = sorted(matches_CC, key=lambda x: x.distance)
        return good_sorted
    
    elif method_post == 'lowe':
        # Apply Lowe's ratio test
        threshold = 0.75

        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append([m])
        
        good_matches_sorted = sorted(good_matches, key=lambda x: x[0].distance)
        good_kp_pairs = [(kp1[match[0].queryIdx], kp2[match[0].trainIdx]) for match in good_matches_sorted]
        return good_kp_pairs, good_matches_sorted, kp1, kp2

    else:
        return [(kp1[match[0].queryIdx], kp2[match[0].trainIdx]) for match in matches], matches
    
def draw_good_keypoints(im1, im2, good_keypoints_1, good_keypoints_2, good_matches, nb_matches_to_draw):
    # cv.drawMatchesKnn expects list of lists as matches.

    img3 = cv.drawMatchesKnn(
        im1,
        good_keypoints_1,
        im2,
        good_keypoints_2,
        good_matches[:nb_matches_to_draw],
        outImg=None,
        singlePointColor=(255, 0, 0),
    )
    plt.figure(figsize=(10, 5))
    plt.imshow(img3)
    plt.axis("off")
    plt.show()


#TODO : Implement keypoint filtering with coded Blender functions
def kp_filter(kp_filename_1, kp_filename_2, idxs):
    kps_im_1 = load_keypoints(kp_filename_1)
    kps_im_2 = load_keypoints(kp_filename_2)
    filtered_kps_im_1 = [kps_im_1[i] for i in idxs]
    filtered_kps_im_2 = [kps_im_2[i] for i in idxs]
    return filtered_kps_im_1, filtered_kps_im_2
    
if __name__ == '__main__':
    save_keypoint_pairs(get_keypoint_pairs('../data/blender/demo_cube_l.png', '../data/blender/demo_cube_r.png')[0], '../data/blender/keypoints')