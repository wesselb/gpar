import numpy as np
from gpar import GPARRegressor, log_transform
from lab import B

B.epsilon = 1e-8


def load(fp):
    data = np.genfromtxt(fp, delimiter=',', dtype=str)[:, 1:]
    header, data = data[0], data[1:].astype(float)
    header = [h[1:-1] for h in header]  # Remove quotes.
    x = data[:, [header.index(name) for name in ['Xloc', 'Yloc']]]
    y = data[:, [header.index(name) for name in ['Ni', 'Zn', 'Cd']]]
    return x, y


# Load and extract data.
x_train, y_train = load('examples/data/jura/jura_prediction.dat')
x_test, y_test = load('examples/data/jura/jura_validation.dat')

# Append first two outputs of test data to training data: the last one is
# predicted.
x_train = np.concatenate((x_train, x_test), axis=0)
y_train_test = y_test.copy()
y_train_test[:, -1] = np.nan
y_train = np.concatenate((y_train, y_train_test), axis=0)

# Fit and predict GPAR.
model = GPARRegressor(scale=10.,
                      linear=False, nonlinear=True, nonlinear_scale=1.0,
                      noise=0.1,
                      impute=True, replace=True, normalise_y=True,
                      transform_y=log_transform)
model.fit(x_train, y_train, fix=False)
means_test = model.predict(x_test, num_samples=200, latent=True)

# Compute MAE.
print('MAE:', np.nanmean(np.abs(y_test[:, -1] - means_test[:, -1])))
