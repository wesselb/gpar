import pandas as pd
import wbml.metric
import wbml.out
from wbml.data.jura import load
from wbml.experiment import WorkingDirectory

from gpar import GPARRegressor, log_transform


def inputs(df):
    return df.reset_index()[["x", "y"]].to_numpy()


if __name__ == "__main__":
    wbml.out.report_time = True
    wd = WorkingDirectory("_experiments", "jura")

    train, test = load()

    # Fit and predict GPAR.
    model = GPARRegressor(
        scale=10.0,
        linear=False,
        nonlinear=True,
        nonlinear_scale=1.0,
        noise=0.1,
        impute=True,
        replace=True,
        normalise_y=True,
        transform_y=log_transform,
    )
    model.fit(inputs(train), train.to_numpy(), fix=False)
    means = model.predict(inputs(test), num_samples=200, latent=True)
    means = pd.DataFrame(means, index=test.index, columns=train.columns)

    wbml.out.kv("MAE", wbml.metric.mae(means, test)["Cd"])
