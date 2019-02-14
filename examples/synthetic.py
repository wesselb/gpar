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
                      linear_slope=0.1,
                      nonlinear_scale=0.1,
                      noise=0.1,
                      impute=True,
                      replace=True)
model.fit(x_obs, y_obs)
means, lowers, uppers = model.predict(x, num_samples=200, credible_bounds=True)

# Plot the result.
plt.figure(figsize=(10, 5))

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.title('Output {}'.format(i + 1))
    plt.scatter(x_obs, y_obs[:, i], label='Observations', c='black')
    plt.plot(x, f[:, i], label='Truth', c='tab:orange')
    plt.plot(x, means[:, i], label='Prediction', c='tab:blue')
    plt.plot(x, lowers[:, i], c='tab:blue', ls='--')
    plt.plot(x, uppers[:, i], c='tab:blue', ls='--')
    plt.legend()

plt.show()
