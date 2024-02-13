import bpy
import sys
import numpy as np

# current working directory corresponds to the one of the virtual scene
# therefore in data/blender/rocks for instance

# add the path to the filtering script
sys.path.append("../../../blender")
from line import check_correct_match_pt
from shift import img_px_to_img_m, get_cam_parameters
from mathutils import Vector


def check_correct_match(kp_arr_file, cam_1, cam_2, epsilon=None):
    """Checks the validity of pairs of matched keypoints in stereo imagery

    Returns :
    (correct_matches, matched_points) where
    - correct_matches (list of tuples) contains tuples of correctly matched pairs of keypoint coordinates
    - matched_points (list of Vector) contains the coordinates of the corresponding 3D points
    """

    # Get camera parameters
    params_cam_1 = get_cam_parameters(cam_1)
    params_cam_2 = get_cam_parameters(cam_2)
    scene = bpy.context.scene

    # pt_pairs is a list of pairs of OpenCV keypoints, we need to get coordinates from it
    kp_arr = np.load(kp_arr_file)
    image_distances = np.empty(kp_arr.shape[1]) # image_distances[i] is the distance between the two points in matches[i]
    correct_matches = []
    correct_matches_idxs = []
    matched_3d_pts = []
    bpy.context.view_layer.update()

    for i in range(kp_arr.shape[1]):

        x1_cv_px = round(kp_arr[0, i, 0])  # careful : rounding
        y1_cv_px = round(kp_arr[0, i, 1])

        x2_cv_px = round(kp_arr[1, i, 0])
        y2_cv_px = round(kp_arr[1, i, 1])

        r, vec, dist = check_correct_match_pt(
            scene,
            x1_cv_px,
            y1_cv_px,
            x2_cv_px,
            y2_cv_px,
            cam_1,
            params_cam_1,
            cam_2,
            params_cam_2,
            epsilon,
        )

        if r:
            correct_matches.append(
                ((x1_cv_px, y1_cv_px), (x2_cv_px, y2_cv_px))
            )  # WARNING : Px coordinates used here, will not work if you try to show the points in Blender
            print(((x1_cv_px, y1_cv_px), (x2_cv_px, y2_cv_px)))
            correct_matches_idxs.append(i)
            matched_3d_pts.append(vec)

        image_distances[i] = dist
        

    return correct_matches, correct_matches_idxs, matched_3d_pts, image_distances


def save_correct_matches(correct_matches_idxs, idx_filename):
    """
    - correct_matches_idxs (list of int)
    - idx_filename (str)
    """
    np.save(idx_filename, np.array(correct_matches_idxs))
