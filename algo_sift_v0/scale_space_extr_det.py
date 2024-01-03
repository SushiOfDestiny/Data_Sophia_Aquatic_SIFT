import numpy as np
import scipy.signal as sig


def gaussian_kernel2D(sigma, shape=None):
    """
    Compute 2D gaussian kernel of coordinates
    of a given shape and standard deviation sigma
    """

    if shape is None:
        shape = (int(6 * sigma + 1), int(6 * sigma + 1))
        # heuristic for the shape

    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2))
        * np.exp(
            -((x - (shape[0] - 1) / 2) ** 2 + (y - (shape[1] - 1) / 2) ** 2)
            / (2 * sigma**2)
        ),
        shape,
    )

    return kernel / np.sum(kernel)
