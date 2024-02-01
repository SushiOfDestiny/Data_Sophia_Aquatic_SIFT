import bpy
import sys
import numpy as np
sys.path.append('../../../blender')
from line import check_correct_match_pt
from shift import img_px_to_img_m
from mathutils import Vector

def check_correct_match(objs, kp_arr_file, cam_1, cam_2, epsilon=None):
    '''Checks the validity of pairs of matched keypoints in stereo imagery
    
    Returns :
    (correct_matches, matched_points) where 
    - correct_matches (list of tuples) contains tuples of correctly matched pairs of keypoint coordinates
    - matched_points (list of Vector) contains the coordinates of the corresponding 3D points'''

    # Get camera parameters
    params_cam_1 = get_cam_parameters(cam_1)
    params_cam_2 = get_cam_parameters(cam_2)
    scene = bpy.context.scene

    # pt_pairs is a list of pairs of OpenCV keypoints, we need to get coordinates from it
    kp_arr = np.load(kp_arr_file)
    correct_matches = []
    correct_matches_idxs = []
    matched_3d_pts = []
    bpy.context.view_layer.update()

    for i in range(kp_arr.shape[1]):

        # Careful with coordinates (x is second dimension in opencv images, y first)
        x1_cv_px = round(kp_arr[0, i, 0]) # careful : rounding
        y1_cv_px = round(kp_arr[0, i, 1])

        x2_cv_px = round(kp_arr[1, i, 0])
        y2_cv_px = round(kp_arr[1, i, 1])

        r, vec = check_correct_match_pt(
            objs, x1_cv_px, y1_cv_px, x2_cv_px, y2_cv_px, 
            cam_1, params_cam_1, cam_2, params_cam_2, scene, epsilon
        )
        
        if r:
            correct_matches.append(((x1_cv_px, y1_cv_px), (x2_cv_px, y2_cv_px))) # WARNING : Px coordinates used here, will not work if you try to show the points in Blender
            print(((x1_cv_px, y1_cv_px), (x2_cv_px, y2_cv_px)))
            correct_matches_idxs.append(i)
            matched_3d_pts.append(vec)

    return correct_matches, correct_matches_idxs, matched_3d_pts

def get_cam_parameters(cam):
    camd = cam.data
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_mm = camd.sensor_width
    sensor_height_mm = camd.sensor_height
    aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x

    pixels_in_u_per_mm = resolution_x_in_px * scale / sensor_width_mm
    pixels_in_v_per_mm = resolution_y_in_px * scale * aspect_ratio / sensor_height_mm
    pixel_size_in_u_direction = 1/pixels_in_u_per_mm
    pixel_size_in_v_direction = 1/pixels_in_v_per_mm

    return {'res_x_px': resolution_x_in_px, 
            'res_y_px': resolution_y_in_px,
            'px_size_u_sensor': pixel_size_in_u_direction,
            'px_size_v_sensor': pixel_size_in_v_direction,
            'sensor_width_mm': sensor_width_mm,
            'sensor_height_mm': sensor_height_mm,
            'px_size_u_img_plane': None,
            'px_size_v_img_plane': None}

def save_correct_matches(correct_matches_idxs, idx_filename):
    '''
    - correct_matches_idxs (list of int)
    - idx_filename (str)
'''
    np.save(idx_filename, np.array(correct_matches_idxs))

if __name__ == '__main__':
    print('----------------------------')
    cam_1 = bpy.data.objects['Cam_1']
    cam_2 = bpy.data.objects['Cam_2']

    #epsilon = 0.1 # can be modified later
    correct_matches, correct_matches_idxs, matched_3d_pts = check_correct_match([ob for ob in bpy.data.objects if ob.type == 'MESH'], 'kp_pairs_arr.npy', cam_1, cam_2)

    save_correct_matches(correct_matches_idxs, 'idxs')