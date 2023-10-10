import numpy as np

def Birchfield_Tomasi_Dissimilarity(left_image, right_image, disparity):

    left_image = left_image.astype(np.float32)
    right_image = right_image.astype(np.float32)

    left_image_height, left_image_width = left_image.shape
    right_image_height, right_image_width = right_image.shape

    left_cost_volume = np.zeros((left_image_height, left_image_width, disparity), dtype=np.float32)
    right_cost_volume = np.zeros((right_image_height, right_image_width, disparity), dtype=np.float32)

    for h in range(left_image_height):
        for w in range(left_image_width):
            for d in range(disparity):
                if w - d >= 0:
                    Il = left_image[h, w]
                    Il_next = (left_image[h, w] + left_image[h, w + 1]) / 2 if w + 1 < left_image_width else Il
                    Il_prev = (left_image[h, w - 1] + left_image[h, w]) / 2 if w - 1 >= 0 else Il

                    Ir = right_image[h, w - d]
                    Ir_next = (right_image[h, w - d] + right_image[h, w + 1 - d]) / 2 if w + 1 - d < right_image_width else Ir
                    Ir_prev = (right_image[h, w - 1 - d] + right_image[h, w - d]) / 2 if w - 1 - d >= 0 else Ir

                    Il_min = min(Il, Il_next, Il_prev)
                    Il_max = max(Il, Il_next, Il_prev)
                    Ir_min = min(Ir, Ir_next, Ir_prev)
                    Ir_max = max(Ir, Ir_next, Ir_prev)

                    left_cost = max(0, Il - Ir_max, Ir_min - Il)
                    left_cost_volume[h, w, d] = left_cost

                    right_cost = max(0, Ir - Il_max, Il_min - Ir)
                    right_cost_volume[h, w - d, d] = right_cost

                else:
                    left_cost_volume[h, w, d] = np.inf
                    if 0 <= w - d:
                        right_cost_volume[h, w - d, d] = np.inf

    return left_cost_volume, right_cost_volume