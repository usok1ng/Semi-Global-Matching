import numpy as np
from tqdm import tqdm


def aggregate_cost_volume(cost_volume):
    # TODO: Implement cost volume aggregation

    raise NotImplementedError("aggregate_cost_volume function has not been implemented yet")

    aggregated_costs = None

    forward_pass = list()
    backward_pass = list()

    for idx, (dy, dx) in enumerate(forward_pass):
        # TODO: Implement forward pass
        pass

    for idx, (dy, dx) in enumerate(backward_pass):
        # TODO: Implement backward pass
        pass

    aggregated_volume = np.sum(aggregated_costs, axis=3)
    return aggregated_volume
