import matplotlib.pyplot as plt
import numpy as np
from gpar.regression import GPARRegressor

x = np.linspace(0, 10, 100)
model = GPARRegressor(scale=0.5, nonlinear_scale=0.1, replace=True)
sample = model.sample(x, p=4)
model.fit(x, sample)
means = model.predict(x)
lowers, uppers = model.lowers, model.uppers

# Plot the result.
plt.figure(figsize=(10, 5))

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.title('Output {}'.format(i + 1))
    plt.scatter(x, sample[:, i], label='Observations', c='tab:green')
    plt.plot(x, means[:, i], label='Prediction', c='tab:red')
    plt.plot(x, lowers[:, i], c='tab:red', ls='--')
    plt.plot(x, uppers[:, i], c='tab:red', ls='--')
    plt.legend()

plt.show()
