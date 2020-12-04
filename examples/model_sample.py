import matplotlib.pyplot as plt
import numpy as np
import wbml.plot

from gpar.regression import GPARRegressor

x = np.linspace(0, 1, 100)
model = GPARRegressor(
    scale=0.1,
    linear=False,
    nonlinear=True,
    nonlinear_scale=0.5,
    impute=True,
    replace=True,
    noise=0.1,
    normalise_y=True,
)

# Sample observations and discard some.
y = model.sample(x, p=3)
y_obs = y.copy()
y_obs[np.random.permutation(100)[:25], 0] = np.nan
y_obs[np.random.permutation(100)[:50], 1] = np.nan
y_obs[np.random.permutation(100)[:75], 2] = np.nan

# Fit model and predict.
model.fit(x, y)
means, lowers, uppers = model.predict(
    x, num_samples=200, latent=False, credible_bounds=True
)

# Plot the result.
plt.figure(figsize=(8, 6))

for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(x, means[:, i], label="Prediction", style="pred")
    plt.fill_between(x, lowers[:, i], uppers[:, i], style="pred")
    plt.scatter(x, y[:, i], label="Truth", style="test")
    plt.scatter(x, y_obs[:, i], label="Observations", style="train")
    plt.ylabel(f"Output {i + 1}")
    wbml.plot.tweak(legend=i == 0)

plt.tight_layout()
plt.show()
