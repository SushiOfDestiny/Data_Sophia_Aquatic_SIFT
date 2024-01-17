import bpy
from mathutils import Vector
from draw_points import draw_points

def get_world_from_img_co(x, y, cam):
    D = cam.data.lens / cam.data.sensor_width
    return cam.matrix_world @ Vector((x, y, -D))

def check_ray_intersect(obj, pt_world_1, pt_world_2, cam_1, cam_2, epsilon):
    '''Tests if the rays going from 
    two image points given by their world coordinates
    and the Blender camera centers
    intersect with the geometry at the same point (with epsilon accuracy)
    
    Returns : 
    A tuple (is_intersect, loc) :
    - is_intersect (Bool) is True if the rays intersect, False otherwise
    - loc (Vector | None) is the point of intersection if intersects are at the same point'''

    result_1, vec_1, _, _ = obj.ray_cast(cam_1.location, pt_world_1 - cam_1.location)
    result_2, vec_2, _, _ = obj.ray_cast(cam_2.location, pt_world_2 - cam_2.location)
    if (not result_1) or (not result_2):
        # Check if both rays hit
        return (False, None)
    if (vec_1 - vec_2).length > epsilon:
        # Check if intersect point is the same
        return (False, None)
    return (True, vec_1)

def check_correct_match_pt(objs, x1, y1, x2, y2, cam_1, cam_2, epsilon):
    '''Tests if two image points given by their x and y coordinates are correct stereo matches
    through ray tracing intersection matching
    
    Returns : (is_correct, 3d_loc)'''

    pt_world_1 = get_world_from_img_co(x1, y1, cam_1)
    pt_world_2 = get_world_from_img_co(x2, y2, cam_2)

    for obj in objs:
        r, vec = check_ray_intersect(obj, pt_world_1, pt_world_2, cam_1, cam_2, epsilon)
        if r:
            return (r, vec)
    
    # If no ray hits have been found, return incorrect match
    return (False, None)