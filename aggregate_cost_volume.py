import numpy as np
from tqdm import tqdm

def aggregate_cost_volume(cost_volume):
    height, width, d = cost_volume.shape
    aggregated_costs = np.zeros([height, width, d])
    ## aggregated_costs = np.zeros_like(cost_volume)

    forward_pass = [(0, 1), (1, 0)]
    backward_pass = [(0, -1), (-1, 0)]

    for idx, (dy, dx) in enumerate(forward_pass):
        for y in range(height):
            for x in range(width):
                for disparity in range(d):
                    aggregated_costs[y, x, disparity] = cost_volume[y, x, disparity]
                    if y - dy >= 0 and x - dx >= 0:
                        aggregated_costs[y, x, disparity] += np.min(aggregated_costs[y - dy, x - dx])

    for idx, (dy, dx) in enumerate(backward_pass):
        for y in reversed(range(height)):
            for x in reversed(range(width)):
                for disparity in range(d):
                    if y + dy < height and x + dx < width:
                        aggregated_costs[y, x, disparity] += np.min(aggregated_costs[y + dy, x + dx])

    aggregated_volume = np.sum(aggregated_costs, axis=2)
    return aggregated_volume