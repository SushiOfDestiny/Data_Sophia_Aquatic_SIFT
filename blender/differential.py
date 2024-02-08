# Archiving purposes

'''
if epsilon is None:
        # Compute projected 3D 1px shifted image points
        x1_img_m_1px_shift, y1_img_m_1px_shift = img_px_to_img_m((x1_cv_px + 1)%scene.render.resolution_x, y1_cv_px, cam_1_params, cam_1, scene)
        x2_img_m_1px_shift, y2_img_m_1px_shift = img_px_to_img_m((x2_cv_px + 1)%scene.render.resolution_x, y2_cv_px, cam_2_params, cam_2, scene)

        pt_world_1_1px_shift = get_world_from_img_co(x1_img_m_1px_shift, y1_img_m_1px_shift, cam_1)
        pt_world_2_1px_shift = get_world_from_img_co(x2_img_m_1px_shift, y2_img_m_1px_shift, cam_2)

        result_1_1px_shift, vec_1_1px_shift, _, _, _, _ = scene.ray_cast(depsgraph, cam_1.location, pt_world_1_1px_shift - cam_1.location)
        result_2_1px_shift, vec_2_1px_shift, _, _, _, _ = scene.ray_cast(depsgraph, cam_2.location, pt_world_2_1px_shift - cam_2.location)

        # Compute distance between projected 3D points and 3D projections of 1px shifted image points
        d1 = (vec_1 - vec_1_1px_shift).length
        d2 = (vec_2 - vec_2_1px_shift).length
'''

import bpy
import bpy_extras
import numpy as np
from shift import img_px_to_world_co, get_cam_parameters
from draw_points import draw_points

def get_depth_at_world_coords(vec_img_pt_world, cam, scene):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    res, vec, _, _, _, _ = scene.ray_cast(depsgraph, cam.location, vec_img_pt_world - cam.location)
    if not res:
        return 0
    return bpy_extras.object_utils.world_to_camera_view(scene, cam, vec)[2] - (cam.data.lens / cam.data.sensor_width)
# Takes into account shift from image plane, but behavior of world_to_camera_view needs to be tested

def get_depth_at_px_coords(x_px_cv, y_px_cv, cam, scene):
    return get_depth_at_world_coords(img_px_to_world_co(x_px_cv, y_px_cv, cam, scene), cam, scene)

def compute_depth_map(cam, scene):
    res_x_px = scene.render.resolution_x
    res_y_px = scene.render.resolution_y
    
    dmap = np.empty((res_y_px, res_x_px), dtype=np.float64)
    for x in range(res_y_px):
        for y in range(res_x_px):
            dmap[x, y] = get_depth_at_px_coords(y, x, cam, scene)
    return dmap

if __name__ == '__main__':

    cam_1 = bpy.data.objects['Cam_1']
    scene = bpy.context.scene

    np.save("dmap", compute_depth_map(cam_1, scene))
    print("depth map computed")
