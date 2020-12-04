import pickle

import numpy as np
import wbml.metric
import wbml.out
from wbml.experiment import WorkingDirectory

from gpar import GPARRegressor

if __name__ == "__main__":
    wbml.out.report_time = True
    wd = WorkingDirectory("_experiments", "ml")

    # Load data.
    with open("examples/paper/ml_data/data.pickle", "rb") as f:
        results = pickle.load(f, encoding="latin1")

    # Generate inputs and outputs.
    output_indices = [0, 5, 10, 15, 20]
    params = results.keys()
    x = np.array([list(p) for p in params])
    y = np.array([np.take(results[p]["val_loss"], output_indices) for p in params])

    # Record number of outputs.
    num_outputs = len(output_indices)

    # Filter extreme data points to reduce noise.
    max_error_at_0 = 5
    min_log_learning_rate = -10
    keep = np.logical_and(x[:, 3] > min_log_learning_rate, y[:, 0] < max_error_at_0)
    x, y = x[keep, :], y[keep, :]

    # Randomly split up into training and testing.
    i_split = int(np.round(0.6 * y.shape[0]))
    perm = np.random.permutation(y.shape[0])
    inds_train, inds_test = perm[:i_split], perm[:i_split]
    x_train, x_test = x[inds_train], x[inds_test]
    y_train, y_test = y[inds_train], y[inds_test]

    # Perform dropping of data.
    prob_drop = 0.3
    indices_all = np.arange(y_train.shape[0])
    indices_remain = indices_all
    for i in range(1, num_outputs):
        # Drop indices randomly.
        n = len(indices_remain)
        perm = np.random.permutation(n)[: int(np.round(0.3 * n))]
        indices_drop = indices_remain[perm]
        indices_remain = np.array(list(set(indices_remain) - set(indices_drop)))

        # Drop data.
        y_train[indices_drop, i:] = np.nan

    # Fit and predict GPAR.
    model = GPARRegressor(
        scale=0.1,
        linear=True,
        linear_scale=100.0,
        nonlinear=True,
        nonlinear_scale=1.0,
        noise=0.01,
        impute=True,
        replace=True,
        normalise_y=True,
    )
    model.fit(x_train, y_train)
    means = model.predict(x_test, num_samples=100, latent=True)

    # Print remaining numbers:
    wbml.out.kv("Remaining", np.sum(~np.isnan(y_train), axis=0))

    # Compute SMSEs for all but the first output.
    wbml.out.kv("SMSE", wbml.metric.smse(means, y_test))
