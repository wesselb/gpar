import sys

import numpy as np
import wbml.out
from lab import B
from wbml.data.air_temp import load as load_temp
from wbml.experiment import WorkingDirectory

from gpar import GPARRegressor


def convert_index(df):
    index = df.index - d_all.index[0]
    return np.array([td.total_seconds() / 3600 / 24 for td in index])


if __name__ == "__main__":
    B.epsilon = 1e-6
    wbml.out.report_time = True
    wd = WorkingDirectory("_experiments", "air_temp")

    # Load data.
    d_size = 0 if len(sys.argv) < 2 else int(sys.argv[1])
    d_all, d_train, d_tests = load_temp()[d_size]

    # Determine the number of inducing points.
    n_ind = [10 * 10 + 1, 10 * 15 + 1, 10 * 31 + 1][d_size]

    # Place inducing points evenly spaced.
    x = convert_index(d_all)
    x_ind = np.linspace(x.min(), x.max(), n_ind)

    # Fit and predict GPAR. NOTE: we use D-GPAR-L-NL here, as opposed to D-GPAR-L,
    # to make the results a little more drastic.
    model = GPARRegressor(
        scale=0.2,
        linear=True,
        linear_scale=10.0,
        nonlinear=True,
        nonlinear_scale=1.0,
        noise=0.1,
        impute=True,
        replace=True,
        normalise_y=True,
        x_ind=x_ind,
    )
    model.fit(convert_index(d_train), d_train.to_numpy())

    # Predict for the test sets.
    preds = []
    for i, d in enumerate(d_tests):
        preds.append(
            model.predict(
                convert_index(d), num_samples=50, credible_bounds=True, latent=False
            )
        )

    # Save predictions.
    wd.save(preds, f"results{d_size}.pickle")
