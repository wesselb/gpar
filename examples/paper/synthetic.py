import matplotlib.pyplot as plt
import numpy as np
from gpar.regression import GPARRegressor

# Create toy data set.
n = 200
x = np.linspace(0, 1, n)
noise = 0.1

# Draw functions depending on each other in complicated ways.
f1 = -np.sin(10 * np.pi * (x + 1)) / (2 * x + 1) - x ** 4
f2 = np.cos(f1) ** 2 + np.sin(3 * x)
f3 = f2 * f1 ** 2 + 3 * x
f = np.stack((f1, f2, f3), axis=0).T

# Add noise and subsample.
y = f + noise * np.random.randn(n, 3)
x_obs, y_obs = x[::8], y[::8]

# Fit and predict GPAR.
model = GPARRegressor(scale=0.1,
                      linear=True, linear_scale=10.,
                      nonlinear=True, nonlinear_scale=0.1,
                      noise=0.1,
                      impute=True, replace=True)
model.fit(x_obs, y_obs)
means, lowers, uppers = \
    model.predict(x, num_samples=100, credible_bounds=True, latent=True)

# Fit and predict independent GPs: set markov=0.
igp = GPARRegressor(scale=0.1,
                    linear=True, linear_scale=10.,
                    nonlinear=True, nonlinear_scale=0.1,
                    noise=0.1, markov=0)
igp.fit(x_obs, y_obs)
igp_means, igp_lowers, igp_uppers = \
    igp.predict(x, num_samples=100, credible_bounds=True, latent=True)

# Plot the result.
plt.figure(figsize=(12, 2.5))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

for i in range(3):
    ax = plt.subplot(1, 3, i + 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.scatter(x_obs, y_obs[:, i], label='Observations', c='black', s=15)
    plt.plot(x, f[:, i], label='Truth', c='tab:orange')
    plt.plot(x, means[:, i], label='GPAR', c='tab:blue')
    plt.fill_between(x, lowers[:, i], uppers[:, i],
                     facecolor='tab:blue', alpha=.25)
    plt.plot(x, igp_means[:, i], label='IGP', c='tab:green')
    plt.fill_between(x, igp_lowers[:, i], igp_uppers[:, i],
                     facecolor='tab:green', alpha=.25)
    plt.xlabel('$t$')
    plt.ylabel('$y_{}$'.format(i + 1))
    if i == 2:
        leg = plt.legend(facecolor='#eeeeee')
        leg.get_frame().set_linewidth(0)

plt.tight_layout()
plt.savefig('examples/paper/synthetic_prediction.pdf')
plt.show()
