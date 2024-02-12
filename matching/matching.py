import cv2 as cv
import matplotlib.pyplot as plt


def get_keypoint_pairs(
    im1,
    im2,
    method_post="lowe",
    contrastThreshold=0.04,
    edgeThreshold=10,
    SIFTsigma=1.6,
    distanceThreshold=0.75,
):
    """

    Call SIFT to detect & compute keypoints and match them between two images.
    im1, im2 : OpenCV int images
    method_post : string, either "lowe" or "crossCheck"
    contrastThreshold : float, threshold to filter out weak keypoints
    edgeThreshold : float, threshold to filter out edge keypoints
    SIFTsigma : float, sigma of the Gaussian applied to the input image at the octave #0
    distanceThreshold : float, threshold for the Lowe's ratio test


    Matching is done through a knn match

    WARNING : CrossCheck is experimental, do not use

    Returns :
    - List of OpenCV keypoints
    - List of OpenCV matches"""

    # Initiate SIFT detector
    sift = cv.SIFT_create(
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        sigma=SIFTsigma,
    )

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    # BFMatcher with default params
    # default norm is L2
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    if matches == []:
        print("No matches found")
        return []

    if method_post == "crossCheck":
        # Alternative: crossCheck=True
        bf_CC = cv.BFMatcher(crossCheck=True)
        matches_CC = bf_CC.match(des1, des2)
        good_sorted = sorted(matches_CC, key=lambda x: x.distance)
        return good_sorted

    elif method_post == "lowe":
        # Apply Lowe's ratio test
        for m, n in matches:
            if m.distance < distanceThreshold * n.distance:
                good_matches.append([m])

        good_matches_sorted = sorted(good_matches, key=lambda x: x[0].distance)
        good_kp_pairs = [
            (kp1[match[0].queryIdx], kp2[match[0].trainIdx])
            for match in good_matches_sorted
        ]
        return good_kp_pairs, good_matches_sorted, kp1, kp2

    else:
        return [
            (kp1[match[0].queryIdx], kp2[match[0].trainIdx]) for match in matches
        ], matches


def draw_good_keypoints(
    im1, im2, good_keypoints_1, good_keypoints_2, good_matches, nb_matches_to_draw
):
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
