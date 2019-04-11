import pickle
import sys

import numpy as np
from gpar import GPARRegressor
from lab import B
from wbml.data.air_temp import load as load_temp

B.epsilon = 1e-6

d_size = 0 if len(sys.argv) < 2 else int(sys.argv[1])
d_all, d_train, d_tests = load_temp()[d_size]
n_ind = [10 * 10 + 1, 10 * 15 + 1, 10 * 31 + 1][d_size]

# Place inducing points evenly spaced.
x_ind = np.linspace(d_all.x[:, 0].min(), d_all.x[:, 0].max(), n_ind)

# Fit and predict GPAR.
#   Note: we use D-GPAR-L-NL here, as opposed to D-GPAR-L, to make the results
#   a little more drastic.
model = GPARRegressor(scale=0.2,
                      linear=True, linear_scale=10.,
                      nonlinear=True, nonlinear_scale=1.,
                      noise=0.1,
                      impute=True, replace=True, normalise_y=True,
                      x_ind=x_ind)
model.fit(d_train.x, d_train.y)

# Predict for the test sets.
preds = []
for i, d in enumerate(d_tests):
    print('Sampling', i + 1)
    preds.append(model.predict(d.x,
                               num_samples=50,
                               credible_bounds=True,
                               latent=False))

# Save predictions.
with open('examples/paper/air_temp_results{}.pickle'.format(d_size), 'wb') as f:
    pickle.dump(preds, f)
