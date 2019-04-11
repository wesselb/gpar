import matplotlib.pyplot as plt
import numpy as np
from gpar.regression import GPARRegressor

x = np.linspace(0, 1, 100)
model = GPARRegressor(scale=0.1,
                      linear=False, nonlinear=True, nonlinear_scale=0.5,
                      impute=True, replace=True,
                      noise=0.1, normalise_y=True)

# Sample observations and discard some.
y = model.sample(x, p=3)
y_obs = y.copy()
y_obs[np.random.permutation(100)[:25], 0] = np.nan
y_obs[np.random.permutation(100)[:50], 1] = np.nan
y_obs[np.random.permutation(100)[:75], 2] = np.nan

# Fit model and predict.
model.fit(x, y)
means, lowers, uppers = \
    model.predict(x, num_samples=200, latent=False, credible_bounds=True)

# Plot the result.
plt.figure(figsize=(8, 6))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

for i in range(3):
    ax = plt.subplot(3, 1, i + 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.ylabel('Output {}'.format(i + 1))
    plt.scatter(x, y[:, i], label='Truth', c='tab:orange')
    plt.scatter(x, y_obs[:, i], label='Observations', c='black')
    plt.plot(x, means[:, i], label='Prediction', c='tab:blue')
    plt.plot(x, lowers[:, i], c='tab:blue', ls='--')
    plt.plot(x, uppers[:, i], c='tab:blue', ls='--')
    if i == 2:
        leg = plt.legend(facecolor='#eeeeee')
        leg.get_frame().set_linewidth(0)

plt.tight_layout()
plt.savefig('examples/model_sample_prediction.pdf')
plt.show()
