import numpy as np
from tqdm import tqdm

def warp_image(reference, img, disparity_map, direction = 'left'):
    height, width, channels = reference.shape
    warped_image = np.copy(reference)

    for h in range(height):
        for w in range(width):
            if direction == 'left':
                new_w = int(w + disparity_map[h, w])
                if 0 <= new_w < width:
                    warped_image[h, w, :] = img[h, new_w, :]
            else:
                new_w = int(w - disparity_map[h, w])
                if 0 <= new_w < width:
                    warped_image[h, w, :] = img[h, new_w, :]
    return warped_image
