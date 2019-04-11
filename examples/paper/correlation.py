import matplotlib.pyplot as plt
import numpy as np
from gpar.regression import GPARRegressor

# Create toy data set.
n = 100
x = np.linspace(0, 1, n)

# Construct complicated functions.
f1 = -np.sin(10 * np.pi * (x + 1)) / (2 * x + 1) - x ** 4
f2 = (0.572 * np.cos(5.334 * np.pi * x) +
      0.0218 * np.cos(24.44 * np.pi * x)) * np.exp(2 * x) / 5. + (2 * x) ** .5
f = np.stack((f1, f2), axis=0).T


# Construct correlated noises.
def draw_ns():
    e1, e2 = 0.05 ** .5 * np.random.randn(2, n)
    n1 = np.stack((e1,
                   np.sin(2 * np.pi * x) ** 2 * e1 +
                   np.cos(2 * np.pi * x) ** 2 * e2), axis=0).T
    n2 = np.stack((e1, np.sin(np.pi * e1) + e2), axis=0).T
    n3 = np.stack((e1, np.sin(np.pi * x) * e1 + e2), axis=0).T
    return n1, n2, n3


# Construct observed values.
n1, n2, n3 = draw_ns()
y1 = f + n1
y2 = f + n2
y3 = f + n3

# Fit and predict GPAR.
model = GPARRegressor(scale=0.05,
                      linear=False, nonlinear=True, nonlinear_scale=0.001,
                      noise=0.01, impute=False, replace=False)
model.fit(x, y1, trace=True)
samples = model.sample(x, posterior=True, latent=False, num_samples=100)
means = np.mean(samples, axis=0)
deviations = [sample - means for sample in samples]

# Plot the result.
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# Plot the fit.
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(x, means[:, 0])
plt.plot(x, f1)
plt.subplot(1, 2, 2)
plt.plot(x, means[:, 1])
plt.plot(x, f2)
plt.show()

# Plot the correlations.
plt.figure(figsize=(10, 5))
c = np.sin(np.pi * x)
cm = plt.get_cmap('gist_rainbow')
plt.subplot(1, 2, 1)
for ds in deviations:
    plt.scatter(ds[:, 0], ds[:, 1], c=c, cmap=cm, s=5, alpha=.1)
plt.subplot(1, 2, 2)
for _ in range(100):
    n1, _, _ = draw_ns()
    plt.scatter(n1[:, 0], n1[:, 1], c=c, cmap=cm, s=5, alpha=.1)
plt.show()
