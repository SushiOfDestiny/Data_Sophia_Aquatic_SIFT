import bpy

def img_px_to_img_mm(x, y, cam_params):
    '''Shifts and dilates image pixel coordinates
    to go from image pixel to image mm coordinates
    where the origin is at the center of the image plane'''

    res_x_px = cam_params['res_x_px']
    res_y_px = cam_params['res_y_px']
    px_size_u = cam_params['px_size_u']
    px_size_v = cam_params['px_size_v']

    shift_x = res_x_px*px_size_u/2
    shift_y = res_y_px*px_size_v/2
    return ((x*px_size_u - shift_x)/1000, (y*px_size_v - shift_y)/1000)