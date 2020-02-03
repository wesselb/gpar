import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wbml.out
import wbml.plot
from lab import B
from wbml.data.eeg import load
from wbml.experiment import WorkingDirectory

from gpar import GPARRegressor

B.epsilon = 1e-8
wd = WorkingDirectory('_experiments', 'eeg')

_, train, test = load()
y_labels = list(train.columns)

# Convert to NumPy.
x = np.array(train.index)
y_train = np.array(train)
y_test = np.array(test)

# Fit and predict GPAR.
model = GPARRegressor(scale=0.02,
                      linear=False, nonlinear=True, nonlinear_scale=1.0,
                      noise=0.01,
                      impute=True, replace=False, normalise_y=True)
model.fit(x, y_train)
means, lowers, uppers = \
    model.predict(x, num_samples=100, credible_bounds=True, latent=True)

# Compute SMSE.
pred = pd.DataFrame(means, index=train.index, columns=train.columns)
smse = ((pred - test) ** 2).mean().mean() / \
       ((test.mean(axis=0) - test) ** 2).mean().mean()

# Report and save average SMSE.
wbml.out.kv('SMSE', smse)

# Plot the result.
plt.figure(figsize=(12, 9))
wbml.plot.tex()

for i, label in enumerate(y_labels[-3:]):
    y_i = y_labels.index(label)
    ax = plt.subplot(3, 1, i + 1)
    plt.plot(x, means[:, y_i], c='tab:blue')
    plt.fill_between(x, lowers[:, y_i], uppers[:, y_i],
                     facecolor='tab:blue', alpha=.25)
    plt.scatter(train.index, train[label], c='tab:green', marker='x', s=10)
    plt.scatter(test.index, test[label], c='tab:orange', marker='x', s=10)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (volt)')
    plt.title(y_labels[y_i])
    plt.xlim(0.4, 1)
    wbml.plot.tweak(legend=False)

plt.tight_layout()
plt.savefig(wd.file('prediction.pdf'))
plt.show()
