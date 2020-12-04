import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wbml.plot
from wbml.data.exchange import load
from wbml.experiment import WorkingDirectory

from gpar import GPARRegressor

if __name__ == "__main__":
    wbml.out.report_time = True
    wd = WorkingDirectory("_experiments", "exchange")

    _, train, test = load()

    x = np.array(train.index)
    y = np.array(train)

    # Fit and predict GPAR.
    model = GPARRegressor(
        scale=0.1,
        linear=True,
        linear_scale=10.0,
        nonlinear=True,
        nonlinear_scale=1.0,
        rq=True,
        noise=0.01,
        impute=True,
        replace=False,
        normalise_y=True,
    )
    model.fit(x, y)
    means, lowers, uppers = model.predict(
        x, num_samples=200, credible_bounds=True, latent=False
    )

    # For the purpose of comparison, standardise using the mean of the *training*
    # data. This is not how the SMSE usually is defined!
    pred = pd.DataFrame(means, index=train.index, columns=train.columns)
    mse = ((pred - test) ** 2).mean(axis=0)
    smse = mse / ((train.mean(axis=0) - test) ** 2).mean(axis=0)

    # Report average SMSE.
    wbml.out.kv("SMSEs", smse.dropna())
    wbml.out.kv("Average SMSE", smse.mean())

    # Plot the result.
    plt.figure(figsize=(15, 3))
    wbml.plot.tex()

    for i, name in enumerate(test.columns):
        p = list(train.columns).index(name)  # Index of output.
        plt.subplot(1, 3, i + 1)
        plt.plot(x, means[:, p], style="pred")
        plt.fill_between(x, lowers[:, p], uppers[:, p], style="pred")
        plt.scatter(x, y[:, p], style="train")
        plt.scatter(test[name].index, test[name], style="test")
        plt.xlabel("Time (year)")
        plt.ylabel(name)
        wbml.plot.tweak(legend=False)

    plt.tight_layout()
    plt.savefig(wd.file("exchange.pdf"))
