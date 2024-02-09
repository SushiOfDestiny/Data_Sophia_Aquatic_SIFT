import bpy, bpy_extras
from shift import img_px_to_img_m, get_world_from_img_co

def check_correct_match_pt(scene, x1_cv_px, y1_cv_px, x2_cv_px, y2_cv_px, cam_1, cam_1_params, cam_2, cam_2_params, epsilon=None):
    '''Tests if two image points given by their x and y coordinates are correct stereo matches
    through ray tracing intersection matching

    It tests if the rays going from 
    two image points given by their world coordinates
    and the Blender camera centers
    intersect with the geometry at the same point (with epsilon accuracy)
    
    Returns : (is_correct, 3d_loc)'''

    # Get image frame coordinates from pixel coordinates
    x1_img_m, y1_img_m = img_px_to_img_m(x1_cv_px, y1_cv_px, cam_1, scene)
    x2_img_m, y2_img_m = img_px_to_img_m(x2_cv_px, y2_cv_px, cam_2, scene)

    # Compute world coordinates from image frame coordinates
    pt_world_1 = get_world_from_img_co(x1_img_m, y1_img_m, cam_1)
    pt_world_2 = get_world_from_img_co(x2_img_m, y2_img_m, cam_2)

    # Compute dependecy graph for scene.ray_cast()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Compute rays
    result_1, vec_1, _, _ , _, _ = scene.ray_cast(depsgraph, cam_1.location, pt_world_1 - cam_1.location)
    result_2, vec_2, _, _, _, _ = scene.ray_cast(depsgraph, cam_2.location, pt_world_2 - cam_2.location)

    pt_world_2_on_img1_camco = bpy_extras.object_utils.world_to_camera_view(scene, cam_1, vec_2)
    x2_img1_camco, y2_img1_camco = pt_world_2_on_img1_camco[0], pt_world_2_on_img1_camco[1]
    # Dilate and shift coordinates
    x2_img1_px = x2_img1_camco*cam_1_params['res_x_px']
    y2_img1_px = (cam_1_params['res_y_px']- y2_img1_camco*cam_1_params['res_y_px'])

    if epsilon is None:
        if abs(x2_img1_px - x1_cv_px) >= 1 or abs(y2_img1_px - y1_cv_px) >= 1:
            return False, None
    else:
        if (vec_1 - vec_2).length > epsilon:
            return False, None
    
    print("Match")
    return True, vec_1

    # if (not result_1) or (not result_2):
    #     # Check if both rays hit
    #     return (False, None)

    # if epsilon is None:
    #     # Compute projected 3D 1px shifted image points
    #     x1_img_m_1px_shift, y1_img_m_1px_shift = img_px_to_img_m((x1_cv_px + 1)%scene.render.resolution_x, y1_cv_px, cam_1, scene)
    #     x2_img_m_1px_shift, y2_img_m_1px_shift = img_px_to_img_m((x2_cv_px + 1)%scene.render.resolution_x, y2_cv_px, cam_2, scene)

    #     pt_world_1_1px_shift = get_world_from_img_co(x1_img_m_1px_shift, y1_img_m_1px_shift, cam_1)
    #     pt_world_2_1px_shift = get_world_from_img_co(x2_img_m_1px_shift, y2_img_m_1px_shift, cam_2)

    #     result_1_1px_shift, vec_1_1px_shift, _, _, _, _ = scene.ray_cast(depsgraph, cam_1.location, pt_world_1_1px_shift - cam_1.location)
    #     result_2_1px_shift, vec_2_1px_shift, _, _, _, _ = scene.ray_cast(depsgraph, cam_2.location, pt_world_2_1px_shift - cam_2.location)

    #     # Compute distance between projected 3D points and 3D projections of 1px shifted image points
    #     d1 = (vec_1 - vec_1_1px_shift).length
    #     d2 = (vec_2 - vec_2_1px_shift).length

    #     epsilon = min((d1, d2)) #min or max ?
    #     print("Epsilon = "+str(epsilon))

    # if (vec_1 - vec_2).length < epsilon:
    #     return (True, vec_1)
    
    # return (False, None)
    