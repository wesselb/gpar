import numpy as np
import matplotlib.pyplot as plt

from gpar import GPAR, Data

from stheno import GP, EQ

x = np.linspace(0, 10, 100)[:, None]


def gp_eq():
    return GP(EQ() > 0.5), 0.1


# Construct a GPAR model.
gpar = GPAR()
gpar = gpar.add_layer(gp_eq)
gpar = gpar.add_layer(gp_eq)

# Sample GPAR.
sample = gpar.sample(x)

# Condition on a sample.
gpar |= Data(x, sample)

# Predict by sampling.
samples = [gpar.sample(x) for _ in range(50)]
means = np.mean(samples, axis=0)
lowers = np.percentile(samples, 2.5, axis=0)
uppers = np.percentile(samples, 100 - 2.5, axis=0)

# Plot the result.
x = x[:, 0]
plt.figure(figsize=(10, 5))

for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.title('Output {}'.format(i + 1))
    plt.scatter(x, sample[:, i], label='Observations', c='tab:green')
    plt.plot(x, means[:, i], label='Prediction', c='tab:red')
    plt.plot(x, lowers[:, i], c='tab:red', ls='--')
    plt.plot(x, uppers[:, i], c='tab:red', ls='--')
    plt.legend()

plt.show()
