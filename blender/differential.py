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