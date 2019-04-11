import pickle

import matplotlib.pyplot as plt
import numpy as np
from gpar import GPARRegressor
from lab import B

B.epsilon = 1e-8

# Load and extract data.
with open('examples/data/eeg/experiment.pickle', 'rb') as f:
    trial = pickle.load(f)
x = trial['x']
y_train = trial['y_train']
y_test = trial['y_test']
y_labels = trial['y_labels']

# Fit and predict GPAR.
model = GPARRegressor(scale=0.02,
                      linear=False, nonlinear=True, nonlinear_scale=1.0,
                      noise=0.01,
                      impute=True, replace=False, normalise_y=True)
model.fit(x, y_train, trace=True)
means, lowers, uppers = \
    model.predict(x, num_samples=100, credible_bounds=True, latent=True)

# Compute SMSE.
i_test = np.any(~np.isnan(y_test), axis=0)
mse_mean = np.nanmean((y_test[:, i_test] - np.nanmean(y_test[:, i_test],
                                                      axis=0, keepdims=True))
                      ** 2)
mse_gpar = np.nanmean((y_test[:, i_test] - means[:, i_test]) ** 2)
print('SMSE:', mse_gpar / mse_mean)

# Plot the result.
plt.figure(figsize=(12, 9))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

for i, y_i in enumerate(range(len(y_labels) - 3, len(y_labels))):
    ax = plt.subplot(3, 1, i + 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.plot(x, means[:, y_i], c='tab:blue')
    plt.fill_between(x, lowers[:, y_i], uppers[:, y_i],
                     facecolor='tab:blue', alpha=.25)
    plt.scatter(x, y_train[:, y_i], c='tab:green', marker='x', s=10)
    plt.scatter(x, y_test[:, y_i], c='tab:orange', marker='x', s=10)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (volt)')
    plt.title(y_labels[y_i])
    plt.xlim(0, 1)

plt.tight_layout()
plt.savefig('examples/paper/eeg_prediction.pdf')
plt.show()
