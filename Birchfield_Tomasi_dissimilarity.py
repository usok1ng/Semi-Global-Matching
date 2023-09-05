import numpy as np


def Birchfield_Tomasi_dissimilarity(left_image, right_image, d):
    # TODO: Implement Birchfield-Tomasi dissimilarity
    # Hint: Fill undefined elements with np.inf at the end

    raise NotImplementedError("Birchfield_Tomasi_dissimilarity function has not been implemented yet")

    left_cost_volume = None
    right_cost_volume = None

    left_disparity = left_cost_volume.argmin(axis=2)
    right_disparity = right_cost_volume.argmin(axis=2)

    return left_cost_volume, right_cost_volume, left_disparity, right_disparity
