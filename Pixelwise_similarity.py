import numpy as np

def SAD(left_image, right_image, disparity):
    # TODO : Implement SAD (Sum of Absolute Differences)

    raise NotImplementedError("SAD function has not been implemented yet")

    left_image_height, left_image_width = left_image.shape
    right_image_height, right_image_width = right_image.shape
    
    cost_volume = np.zeros((left_image_height, left_image_width, disparity), dtype=np.float32)

    for h in range(left_image_height):
        for w in range(left_image_width):
            for d in range(disparity):
                if w + d < right_image_width:
                    cost_volume[h][w][d] = np.abs(left_image[h][w] - right_image[h][w + d])
                else:
                    cost_volume[h][w][d] = np.inf

    disparity_map = cost_volume.argmin(axis=2)

    return cost_volume, disparity_map

def SSD(left_image, right_image, disparity):
    # TODO : Implement SSD (Sum of Squared Differences)

    raise NotImplementedError("SSD function has not been implemented yet")

    left_image_height, left_image_width = left_image.shape
    right_image_height, right_image_width = right_image.shape
    
    cost_volume = np.zeros((left_image_height, left_image_width, disparity), dtype=np.float32)

    for h in range(left_image_height):
        for w in range(left_image_width):
            for d in range(disparity):
                if w + d < right_image_width:
                    cost_volume[h][w][d] = np.power((left_image[h][w] - right_image[h][w + d]), 2)
                else:
                    cost_volume[h][w][d] = np.inf

    disparity_map = cost_volume.argmin(axis=2)

    return cost_volume, disparity_map