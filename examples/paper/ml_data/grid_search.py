import os
import pickle
import random

import numpy as np
from evaluate_neural_net import get_results
from sklearn.model_selection import ParameterGrid

if __name__ == "__main__":
    if os.path.isfile("data.pickle"):
        with open("data.pickle", "rb") as handle:
            results = pickle.load(handle)
    else:
        results = {}

    params_max_min_step = {
        "n_neurons": {"min": 50.0, "max": 500.0, "step_size": 50.0},
        "n_hidden_layers": {"min": 1.0, "max": 3.0, "step_size": 1.0},
        "prob_drop_out": {"min": 0.0, "max": 0.9, "step_size": 0.1},
        "log_learning_rate": {"min": -20.0, "max": 0.0, "step_size": 1.0},
        "log_l1_weight_reg": {"min": -20.0, "max": 0.0, "step_size": 1.0},
        "log_l2_weight_reg": {"min": -20.0, "max": 0.0, "step_size": 1.0},
    }

    param_grid = {
        key: list(
            np.arange(
                params_max_min_step[key]["min"],
                params_max_min_step[key]["max"],
                params_max_min_step[key]["step_size"],
            )
        )
        for key in params_max_min_step
    }

    grid = list(ParameterGrid(param_grid=param_grid))
    random.shuffle(grid)

    key_order = [
        "n_neurons",
        "n_hidden_layers",
        "prob_drop_out",
        "log_learning_rate",
        "log_l1_weight_reg",
        "log_l2_weight_reg",
    ]

    for param in grid:
        k = tuple([param[key] for key in key_order])
        if k not in results.keys():
            results[k] = get_results(param)

            with open("data.pickle", "wb") as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
