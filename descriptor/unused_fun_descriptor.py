####################
# Unused functions #
####################


def stack_features(eigvals, eigvects, gradients, orientations):
    """
    Return list of the features arrays, in right order
    """
    features = [
        eigvals[:, :, 0],
        eigvals[:, :, 1],
        eigvects[:, :, 0],
        eigvects[:, :, 1],
        gradients,
        orientations,
    ]
    return features


def compute_compact_features_vect(features, position):
    """
    features: list of separated features arrays for all pixels: eigvals1, eigvals2, eigvects1, eigvects2, gradients, orientations
    Return: compact features vector of position (x, y)
    """
    x, y = position
    return [feature[y, x] for feature in features]


def flat2compact_features_vect(flat_features_vect):
    eigval1 = flat_features_vect[0]
    eigval2 = flat_features_vect[1]
    eigvect1 = flat_features_vect[2:4]
    eigvect2 = flat_features_vect[4:6]
    gradient = flat_features_vect[6:8]
    orientation = flat_features_vect[8]
    return [eigval1, eigval2, eigvect1, eigvect2, gradient, orientation]


def compact2flat_features_vect(compact_features_vect):
    """
    Convert a list of compact features of a keypoint into a flat feature vector,
    meaning 1 coef that is a 2D vector in the compact features vector is split into 2 coefs in the flat features vector.
    """
    flat_features = []
    for feature in compact_features_vect:
        flat_features.extend(feature.flatten())
    return flat_features


def compute_raw_features_vect_v2(features, position, radius=2, sigma=1.6):
    """
    Compute gaussian means of multiple features around a keypoint, and stack them in a vector.
    Warning: features that are already vectors (gradients, eigenvectors) are put intact in the vector.
    features: gradients, orientations, first eigenvalues, second eigenvalues, first eigenvectors, second eigenvectors of all pixels of an image, float32 arrays
    """
    x, y = position
    # crop subfeatures around the keypoint
    subfeatures = [
        feature[y - radius : y + radius + 1, x - radius : x + radius + 1]
        for feature in features
    ]
    # compute gaussian means of subfeatures
    kp_features = [
        compute_gaussian_mean(subfeature, sigma) for subfeature in subfeatures
    ]
    return kp_features


def compute_descriptors_img_v2(g_img, border_size=1, kp_radius=2, sigma=1.6):
    """
    Compute descriptors for all pixels of a grayscale image, each considered as a keypoint
    g_img: float32 grayscale image
    border_size: int size of the border to ignore for feature computation (default is 1)
    kp_radius: int radius of the subfeatures to consider to calculate averaged features of a keypoint (default is 2)
    sigma: float sigma of the Gaussian used to compute the gaussian mean of the features (default is 1.6)
    """
    # precompute individual features of all pixels
    eigvals, eigvects, gradients = vh.compute_hessian_gradient_subimage(
        g_img, border_size
    )
    orientations = compute_orientations(g_img, border_size)
    features = stack_features(eigvals, eigvects, gradients, orientations)
    len_features_vect = len(features)

    # define the borders of pixels considered as keypoints with the border of feature and the desired radius around the keypoint
    kp_border_size = border_size + kp_radius
    # define array of features of all keypoints, indexed by keypoint position
    # in positions of non keypoints pixels, the feature vector is filled with 0
    kp_features = np.zeros(
        (g_img.shape[0], g_img.shape[1], len_features_vect), dtype=np.float32
    )

    # compute descriptors for all keypoints:
    for y_kp in range(kp_border_size, g_img.shape[0] - kp_border_size):
        for x_kp in range(kp_border_size, g_img.shape[1] - kp_border_size):
            kp_features


# def compute_features_overall(g_img, border_size=1):
#     """
#     Compute useful features for all pixels of a grayscale image.
#     Return a list of pairs of features arrays, each pair containing the feature values and the feature orientations.
#     All angular features are in degrees in [0, 360[.
#     """
#     # compute eigenvalues and eigenvectors of the Hessian matrix
#     eigvals, eigvects, gradients = vh.compute_hessian_gradient_subimage(
#         g_img, border_size
#     )
#     # compute gradients norms
#     gradients_norms = np.linalg.norm(gradients, axis=2)

#     # compute principal directions of the Hessian matrix
#     principal_directions = compute_horiz_angles(eigvects)

#     # compute orientations of the gradients in degrees
#     orientations = compute_orientations(g_img, border_size)

#     # convert and rescale angles in [0, 360[
#     posdeg_orientations = convert_angles_to_pos_degrees(orientations)
#     posdeg_principal_directions = convert_angles_to_pos_degrees(principal_directions)

#     # separate for each index of eigenvalue, the eigenvalue according to its sign
#     eigvals1pos = eigvals[:, :, 0] * (eigvals[:, :, 0] > 0)
#     prin_dir1pos = posdeg_principal_directions[:, :, 0] * (eigvals[:, :, 0] > 0)
#     eigvals1neg = eigvals[:, :, 0] * (eigvals[:, :, 0] < 0)
#     prin_dir1neg = eigvals[:, :, 0] * (eigvals[:, :, 0] < 0)
#     eigvals2pos = eigvals[:, :, 1] * (eigvals[:, :, 1] > 0)
#     prin_dir2pos = posdeg_principal_directions[:, :, 1] * (eigvals[:, :, 1] > 0)
#     eigvals2neg = eigvals[:, :, 1] * (eigvals[:, :, 1] < 0)
#     prin_dir2neg = eigvals[:, :, 1] * (eigvals[:, :, 1] < 0)

#     # take opposite of negative eigenvalues
#     eigvals1neg = -eigvals1neg
#     eigvals2neg = -eigvals2neg

#     # concatenate the arrays along the last axis
#     eigvalspos = np.concatenate([eigvals1pos, eigvals2pos], axis=0)
#     eigvectspos = np.concatenate([prin_dir1pos, prin_dir2pos], axis=0)
#     eigvalsneg = np.concatenate([eigvals1neg, eigvals2neg], axis=0)
#     eigvectsneg = np.concatenate([prin_dir1neg, prin_dir2neg], axis=0)

#     # stack all features in a list
#     features = [
#         [eigvalspos, eigvectspos],
#         [eigvalsneg, eigvectsneg],
#         [gradients_norms, posdeg_orientations],
#     ]

#     return features


# def rescale_value(values, kp_position):
#     """
#     Rescale vector values by keypoint value.
#     Vector value must be positive.
#     """
#     kp_value = values[kp_position[0], kp_position[1]]
#     return values / kp_value


# def compute_vector_histogram(
#     values,
#     vectors_orientations,
#     kp_position,
#     nb_bins=3,
#     bin_radius=2,
#     delta_angle=5.0,
#     sigma=0,
# ):
#     """
#     Discretize neighborhood of keypoint into nb_bins bins, each of size 2*bin_radius+1, centered on keypoint.
#     For each neighborhood, compute the histogram of the vectors orientations, weighted by vector values and distance to keypoint.
#     Vector value must be positive.
#     nb_bins: int odd number of bins of the neighborhood (default is 3)
#     delta_angle: float size of the angular bins in degrees (default is 5)
#     """
#     # if sigma is null, define it with neighborhood_size
#     neighborhood_size = nb_bins * (2 * bin_radius + 1)
#     if sigma == 0:
#         sigma = neighborhood_size / 2
#     # compute the gaussian weight of the vectors
#     g_weight = np.exp(-1 / (2 * sigma**2))

#     # rescale vectors values by keypoint value
#     rescaled_values = rescale_value(values, kp_position)

#     # rotate vectors orientations with kp orientation in trigo order
#     kp_orientation = vectors_orientations[kp_position]
#     rotated_orientations = (vectors_orientations - kp_orientation) % 360.0

#     # initialize the angular histograms
#     nb_angular_bins = int(360 / delta_angle) + 1
#     histograms = np.zeros((nb_bins, nb_bins, nb_angular_bins), dtype=np.float32)

#     # loop over all pixels and add its contribution to the right angular histogram
#     for bin_j in range(nb_bins):
#         for bin_i in range(nb_bins):
#             # compute the center of the neighborhood
#             y = kp_position[0] + (bin_j - nb_bins // 2) * (2 * bin_radius + 1)
#             x = kp_position[1] + (bin_i - nb_bins // 2) * (2 * bin_radius + 1)
#             # loop over all pixels of the neighborhood
#             for j in range(y - bin_radius, y + bin_radius + 1):
#                 for i in range(x - bin_radius, x + bin_radius + 1):
#                     # compute the angle of the vector
#                     angle = rotated_orientations[j, i]
#                     # compute the gaussian weight of the vector
#                     weight = g_weight * np.exp(
#                         -((j - kp_position[0]) ** 2 + (i - kp_position[1]) ** 2)
#                         / (2 * sigma**2)
#                     )
#                     # compute the bin of the angle
#                     bin_angle = int(angle / delta_angle)
#                     # add the weighted vector to the right histogram
#                     histograms[bin_j, bin_i, bin_angle] += (
#                         weight * rescaled_values[j, i]
#                     )
#     return histograms


# def compute_descriptor_histograms(overall_features, kp_position):
#     """
#     Compute the histograms for the descriptor of a keypoint
#     overall_features: list of features arrays of all pixels of the image
#     kp_position: (x, y) int pixel position of keypoint
#     return descriptor_histograms: list of 3 histograms, each of shape (nb_bins, nb_bins, nb_angular_bins)
#     """
#     descriptor_histograms = []
#     # loop over all features pairs (value, orientation_value), example (gradient norms, gradient orientations)
#     for id_value in range(len(overall_features)):
#         value, value_orientation = overall_features[id_value]
#         descriptor_histograms.append(
#             compute_vector_histogram(
#                 value,
#                 value_orientation,
#                 kp_position,
#             )
#         )
#     return descriptor_histograms
