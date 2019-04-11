import pickle

import numpy as np
from gpar import GPARRegressor
from lab import B

B.epsilon = 1e-8

# Load data.
with open('examples/data/ml/data.pickle', 'rb') as f:
    results = pickle.load(f, encoding='latin1')

# Generate x and y.
output_indices = [0, 5, 10, 15, 20]
params = results.keys()
x = np.array([list(p) for p in params])
y = np.array([np.take(results[p]['val_loss'], output_indices) for p in params])

# Record number of outputs
num_outputs = len(output_indices)

# Filter extreme data points to reduce noise.
max_error_at_0 = 5
min_log_learning_rate = -10
keep = np.logical_and(x[:, 3] > min_log_learning_rate,
                      y[:, 0] < max_error_at_0)
x, y = x[keep, :], y[keep, :]

# Randomly split up into training and testing.
i_split = int(np.round(0.6 * y.shape[0]))
perm = np.random.permutation(y.shape[0])
inds_train, inds_test = perm[:i_split], perm[:i_split]
x_train, x_test = x[inds_train], x[inds_test]
y_train, y_test = y[inds_train], y[inds_test]

# Normalise inputs according to training statistics.
x_mean = np.mean(x_train, axis=0, keepdims=True)
x_std = np.std(x_train, axis=0, keepdims=True)
x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std

# Perform dropping of data.
prob_drop = 0.3
indices_all = np.arange(y_train.shape[0])
indices_remain = indices_all
for i in range(1, num_outputs):
    # Drop indices randomly.
    n = len(indices_remain)
    perm = np.random.permutation(n)[:int(np.round(0.3 * n))]
    indices_drop = indices_remain[perm]
    indices_remain = np.array(list(set(indices_remain) - set(indices_drop)))

    # Drop data.
    y_train[indices_drop, i:] = np.nan

# Fit and predict GPAR.
model = GPARRegressor(scale=0.1,
                      linear=True, linear_scale=100.,
                      nonlinear=True, nonlinear_scale=1.0,
                      noise=0.01,
                      impute=True, replace=True, normalise_y=True)
model.fit(x_train, y_train, trace=True)
means = model.predict(x_test, num_samples=100, latent=True)

# Print remaining numbers:
print('Remaining:', np.sum(~np.isnan(y_train), axis=0))

# Compute SMSEs for all but the first output.
mse_mean = np.nanmean((y_test - np.nanmean(y_test, axis=0, keepdims=True)) ** 2,
                      axis=0)
mse_gpar = np.nanmean((y_test - means) ** 2, axis=0)
print('SMSE:', mse_gpar / mse_mean)
