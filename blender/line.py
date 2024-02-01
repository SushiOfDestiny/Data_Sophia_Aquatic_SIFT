import bpy
from mathutils import Vector
from draw_points import draw_points
from shift import img_px_to_img_m

def get_world_from_img_co(x, y, cam):
    '''Returns world coordinates from image pixel coordinates'''
    D = cam.data.lens / cam.data.sensor_width
    return cam.matrix_world @ Vector((x, y, -D))

def ray_cast_global(obj, start, dir):
    local_start = obj.matrix_world.inverted() @ start
    local_dir = obj.matrix_world.inverted() @ dir
    (res, loc, norm, index) = obj.ray_cast(local_start, local_dir)
    if res:
        location = obj.matrix_world @ loc
        normal = (obj.matrix_world @ norm) - (obj.matrix_world @ Vector((0.0, 0.0, 0.0)))
        return (res, location, normal, index)
    else:
        return (res, loc, norm, index)

def check_ray_intersect(obj, pt_world_1, pt_world_2, cam_1, cam_2, epsilon):
    '''Tests if the rays going from 
    two image points given by their world coordinates
    and the Blender camera centers
    intersect with the geometry at the same point (with epsilon accuracy)
    
    Returns : 
    A tuple (is_intersect, loc) :
    - is_intersect (Bool) is True if the rays intersect, False otherwise
    - loc (Vector | None) is the point of intersection if intersects are at the same point'''

    result_1, vec_1, _, _ = ray_cast_global(obj, cam_1.location, pt_world_1 - cam_1.location)
    result_2, vec_2, _, _ = ray_cast_global(obj, cam_2.location, pt_world_2 - cam_2.location)
    if (not result_1) or (not result_2):
        # Check if both rays hit
        return (False, None)
    if (vec_1 - vec_2).length > epsilon:
        # Check if intersect point is the same or not (if True, not the same)
        return (False, None)

    return (True, vec_1)

def check_correct_match_pt(objs, x1_cv_px, y1_cv_px, x2_cv_px, y2_cv_px, cam_1, cam_1_params, cam_2, cam_2_params, scene, epsilon):
    '''Tests if two image points given by their x and y coordinates are correct stereo matches
    through ray tracing intersection matching
    
    Returns : (is_correct, 3d_loc)'''

    # Get image frame coordinates from pixel coordinates
    x1_img_m, y1_img_m = img_px_to_img_m(x1_cv_px, y1_cv_px, cam_1_params, cam_1, scene)
    x2_img_m, y2_img_m = img_px_to_img_m(x2_cv_px, y2_cv_px, cam_2_params, cam_2, scene)

    if epsilon is None:
        x1_img_m_1px_shift, y1_img_m_1px_shift = img_px_to_img_m((x1_cv_px + 1)%scene.render.resolution_x, y1_cv_px, cam_1_params, cam_1, scene)
        x2_img_m_1px_shift, y2_img_m_1px_shift = img_px_to_img_m((x2_cv_px + 1)%scene.render.resolution_x, y2_cv_px, cam_2_params, cam_2, scene)
        return check_correct_match_pt_variable_epsilon(
            objs, x1_img_m, y1_img_m, x2_img_m, y2_img_m, 
            x1_img_m_1px_shift, y1_img_m_1px_shift, x2_img_m_1px_shift, y2_img_m_1px_shift, 
            cam_1, cam_2, scene)

    pt_world_1 = get_world_from_img_co(x1_img_m, y1_img_m, cam_1)
    pt_world_2 = get_world_from_img_co(x2_img_m, y2_img_m, cam_2)

    for obj in objs:
        r, vec = check_ray_intersect(obj, pt_world_1, pt_world_2, cam_1, cam_2, epsilon)
        if r:
            #draw_points([pt_world_1], 'correct_kp_cam_1')
            #draw_points([pt_world_2], 'correct_kp_cam_2')
            return (r, vec)
    
    # If no ray hits have been found, return incorrect match
    return (False, None)

def check_correct_match_pt_variable_epsilon(objs, x1, y1, x2, y2, x1_shift, y1_shift, x2_shift, y2_shift, cam_1, cam_2, scene):
    pt_world_1 = get_world_from_img_co(x1, y1, cam_1)
    #print("Pt world 1 : "+str(pt_world_1))
    pt_world_2 = get_world_from_img_co(x2, y2, cam_2)

    pt_world_1_1px_shift = get_world_from_img_co(x1_shift, y1_shift, cam_1)
    #print("Pt world 1 1 px shift : "+str(pt_world_1_1px_shift))
    pt_world_2_1px_shift = get_world_from_img_co(x2_shift, y2_shift, cam_2)

    for obj in objs:
        result_1, vec_1, _, _ , _, _ = scene.ray_cast(bpy.context.evaluated_depsgraph_get(), cam_1.location, pt_world_1 - cam_1.location)
        #draw_points([vec_1], 'pt_1')
        result_2, vec_2, _, _, _, _ = scene.ray_cast(bpy.context.evaluated_depsgraph_get(), cam_2.location, pt_world_2 - cam_2.location)
        #draw_points([vec_2], 'pt_2')

        # Check if both rays hit
        if (not result_1) or (not result_2):
            return (False, None)

        # Project neighboring pixels onto 3D scene
        result_1_1px_shift, vec_1_1px_shift, _, _, _, _ = scene.ray_cast(bpy.context.evaluated_depsgraph_get(), cam_1.location, pt_world_1_1px_shift - cam_1.location)
        #draw_points([vec_1_1px_shift], "pt_1_shift")
        result_2_1px_shift, vec_2_1px_shift, _, _, _, _ = scene.ray_cast(bpy.context.evaluated_depsgraph_get(), cam_2.location, pt_world_2_1px_shift - cam_2.location)
        #draw_points([vec_2_1px_shift], "pt_2_shift")

        # Compute distance between projected 3D points and 3D projections of 1px shifted image points
        d1 = (vec_1 - vec_1_1px_shift).length
        d2 = (vec_2 - vec_2_1px_shift).length

        epsilon = max((d1, d2)) #min or max ?
        print("Epsilon = "+str(epsilon))

        print("Distance between ray hits : "+str((vec_1 - vec_2).length))
        if (vec_1 - vec_2).length > 0.5:
            draw_points([vec_1, vec_2, pt_world_1, pt_world_2], 'pt')
        print('--')
        if (vec_1 - vec_2).length < epsilon:
            return (True, vec_1)

    return (False, None)
    