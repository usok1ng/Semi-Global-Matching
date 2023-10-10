import numpy as np


def Birchfield_Tomasi_dissimilarity(left_image, right_image, d):
    # TODO: Implement Birchfield-Tomasi dissimilarity
    # Hint: Fill undefined elements with np.inf at the end

    raise NotImplementedError("Birchfield_Tomasi_dissimilarity function has not been implemented yet")

    left_image_height, left_image_width = left_image.shape
    right_image_height, right_image_width = right_image.shape

    left_cost_volume = np.zeros([left_image.height, left_image.width, d])
    right_cost_volume = np.zeros([right_image.height, right_image.width, d])

    for disparity in range(d):
        for h in range(left_image_height):
            for w in range(left_image_width):
                

    left_disparity = left_cost_volume.argmin(axis=2)
    right_disparity = right_cost_volume.argmin(axis=2)

    return left_cost_volume, right_cost_volume, left_disparity, right_disparity