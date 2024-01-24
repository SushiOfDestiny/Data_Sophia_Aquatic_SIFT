import bpy

def get_camera_keyframe_bounding_points(cam, scene):
    # When using cam.data.view_frame(scene), first point is top right corner,
    # second point is bottom right
    # third point is bottom left
    # fourth point is top left

    # Important to specify scene, otherwise bounding box height is twice the actual size for some reason

    return cam.data.view_frame(scene=scene)

def img_px_to_img_m(x_cv, y_cv, cam_params, cam, scene):
    '''Shifts and dilates image pixel coordinates
    to go from image pixel to image mm coordinates
    where the origin is at the center of the image plane'''

    res_u_px = cam_params['res_x_px']
    res_v_px = cam_params['res_y_px']

    bb_ur, bb_dr, bb_dl, bb_ul = get_camera_keyframe_bounding_points(cam, scene)
    image_plane_width = bb_ur[0] - bb_ul[0]
    image_plane_height = bb_ur[1] - bb_dr[1]

    px_density_u = res_u_px/image_plane_width
    px_size_u = 1.0/px_density_u
    px_density_v = res_v_px/image_plane_height
    px_size_v = 1.0/px_density_v

    # Warning : x direction in OpenCV is u direction here
    # and y direction in OpenCV is -v here

    return ((x_cv - res_u_px/2)*px_size_u, (-y_cv + res_v_px/2)*px_size_v)
    