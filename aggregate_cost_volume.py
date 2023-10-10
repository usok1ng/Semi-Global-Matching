import numpy as np
from tqdm import tqdm

def aggregate_cost_volume(cost_volume):
    height, width, disparity = cost_volume.shape
    aggregated_costs = np.zeros([height, width, disparity])

    directions = [(0, 1), (0, 2), (1, 1), (2, 2), (1, 0), (2, 0), (1, -1), (2, -2), (0, -1), (0, -2), (-1, -1), (-2, -2), (-1, 0), (-2, 0), (-1, 1), (-2, 2)]

    p1 = 5
    p2 = 150
    
    def compute_aggregated_cost(y, x, d):
        new_y = y - dy
        new_x = x - dx
        
        original_cost = cost_volume[y, x, d]

        if 0 <= new_y < height and 0 <= new_x < width:
            cost = aggregated_costs[new_y, new_x, d]
            cost_minus = aggregated_costs[new_y, new_x, d - 1] + p1 if d - 1 >= 0 else np.inf
            cost_plus = aggregated_costs[new_y, new_x, d + 1] + p1 if d + 1 < disparity else np.inf
            min_cost = np.min(aggregated_costs[new_y, new_x, :])
            cost_others = min_cost + p2
            
            return original_cost + min(cost, cost_minus, cost_plus, cost_others) - min_cost
        else:
            return original_cost


    for dy, dx in tqdm(directions):
        # Forward pass
        if (dy, dx) in directions[:8]:
            y_range, x_range = range(height), range(width)
        # Backward pass
        else:
            y_range, x_range = reversed(range(height)), reversed(range(width))

        for y in y_range:
            for x in x_range:
                for d in range(disparity):
                    aggregated_costs[y, x, d] += compute_aggregated_cost(y, x, d)

    return aggregated_costs