import cv2
import numpy as np
from tqdm import tqdm
import os

from Birchfield_Tomasi_Dissimilarity import Birchfield_Tomasi_Dissimilarity
from aggregate_cost_volume import aggregate_cost_volume
from warp import warp_image
from metric import compute_mse, compute_psnr

def left_semi_global_matching(left_image, right_image, disparity):
    left_cost_volume, _ = Birchfield_Tomasi_Dissimilarity(left_image, right_image, disparity)
    aggregated_cost_volume = aggregate_cost_volume(left_cost_volume)
    aggregated_disparity = aggregated_cost_volume.argmin(axis=2)
    return aggregated_disparity

def right_semi_global_matching(left_image, right_image, disparity):
    _, right_cost_volume = Birchfield_Tomasi_Dissimilarity(left_image, right_image, disparity)
    aggregated_cost_volume = aggregate_cost_volume(right_cost_volume)
    aggregated_disparity = aggregated_cost_volume.argmin(axis=2)
    return aggregated_disparity

if __name__ == "__main__":
    input_path = 'input'
    target_path = 'target'
    final_output_path = 'output/Final_Disparity'

    gray_img_list = [cv2.imread(os.path.join(input_path, img), cv2.IMREAD_GRAYSCALE) for img in sorted(os.listdir(input_path))]
    original_img_list = [cv2.imread(os.path.join(input_path, img), cv2.IMREAD_COLOR) for img in sorted(os.listdir(input_path))]
    ground_truth = cv2.imread(os.path.join(target_path, 'gt.png'), cv2.IMREAD_COLOR)

    disparity = 24
    warped_image_list = []

    for idx in [0, 1, 2, 3, 4, 5, 6]:
        if idx < 3:
            disparity_map = right_semi_global_matching(gray_img_list[3], gray_img_list[idx], disparity)
            cv2.imwrite(f'./output/Intermediate_Disparity/disparity_{idx + 1}.png', disparity_map)

            warped_image = warp_image(original_img_list[3], original_img_list[idx], disparity_map, direction = 'left')
            cv2.imwrite(f'./output/Intermediate_Disparity/warped_image_{idx + 1}.png', warped_image)
            warped_image_list.append(warped_image)
        else:
            disparity_map = left_semi_global_matching(gray_img_list[3], gray_img_list[idx], disparity)
            cv2.imwrite(f'./output/Intermediate_Disparity/disparity_{idx + 1}.png', disparity_map)

            warped_image = warp_image(original_img_list[3], original_img_list[idx], disparity_map, direction = 'right')
            cv2.imwrite(f'./output/Intermediate_Disparity/warped_image_{idx + 1}.png', warped_image)
            warped_image_list.append(warped_image)

    boundary_range = disparity

    aggregated_image = np.mean(warped_image_list, axis = 0)
    cropped_aggregated_image = aggregated_image[boundary_range:-boundary_range, boundary_range:-boundary_range]
    cv2.imwrite('./output/Final_Disparity/final_disparity.png', cropped_aggregated_image)

    cropped_ground_truth = ground_truth[boundary_range:-boundary_range, boundary_range:-boundary_range]

    true_image = cropped_ground_truth
    predicted_image = cv2.imread(os.path.join(final_output_path, 'final_disparity.png'), cv2.IMREAD_COLOR)

    mse = compute_mse(true_image, predicted_image)
    print("mse:", mse)

    psnr = compute_psnr(mse)
    print("psnr:", psnr)