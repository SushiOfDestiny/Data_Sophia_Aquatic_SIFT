import bpy
from mathutils import Vector

def get_camera_keyframe_bounding_points(cam, scene):
    # When using cam.data.view_frame(scene), first point is top right corner,
    # second point is bottom right
    # third point is bottom left
    # fourth point is top left

    # Important to specify scene, otherwise bounding box height is twice the actual size for some reason

    return cam.data.view_frame(scene=scene)

def img_px_to_img_m(x_cv, y_cv, cam, scene):
    '''Shifts and dilates image pixel coordinates
    to go from image pixel to image mm coordinates
    where the origin is at the center of the image plane'''

    res_u_px = scene.render.resolution_x
    res_v_px = scene.render.resolution_y

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

def img_m_to_img_px(x_m, y_m, cam, scene):

    res_u_px = scene.render.resolution_x
    res_v_px = scene.render.resolution_y

    bb_ur, bb_dr, bb_dl, bb_ul = get_camera_keyframe_bounding_points(cam, scene)
    image_plane_width = bb_ur[0] - bb_ul[0]
    image_plane_height = bb_ur[1] - bb_dr[1]

    px_density_u = res_u_px/image_plane_width
    px_size_u = 1.0/px_density_u
    px_density_v = res_v_px/image_plane_height
    px_size_v = 1.0/px_density_v

    return (x_m/px_size_u + res_u_px/2, -(y_m/px_size_v - res_v_px/2))

def get_world_from_img_co(x, y, cam):
    '''Returns world coordinates from image meter coordinates'''
    D = cam.data.lens / cam.data.sensor_width
    return cam.matrix_world @ Vector((x, y, -D))

def get_img_from_world_co(vec, cam):
    '''Returns img meter coordinates from world meter coordinates'''
    D = cam.data.lens / cam.data.sensor_width
    return cam.matrix_world.inverted() @ vec

def img_px_to_world_co(x_px_cv, y_px_cv, cam, scene):
    (x_m, y_m) = img_px_to_img_m(x_px_cv, y_px_cv, cam, scene)
    return get_world_from_img_co(x_m, y_m, cam)

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
    pixel_size_in_u_direction = 1 / pixels_in_u_per_mm
    pixel_size_in_v_direction = 1 / pixels_in_v_per_mm

    return {
        "res_x_px": resolution_x_in_px,
        "res_y_px": resolution_y_in_px,
        "px_size_u_sensor": pixel_size_in_u_direction,
        "px_size_v_sensor": pixel_size_in_v_direction,
        "sensor_width_mm": sensor_width_mm,
        "sensor_height_mm": sensor_height_mm,
        "px_size_u_img_plane": None,
        "px_size_v_img_plane": None,
        "distance_to_image_plane": cam.data.lens / cam.data.sensor_width
    }