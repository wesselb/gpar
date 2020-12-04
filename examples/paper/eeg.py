import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wbml.metric
import wbml.out
import wbml.plot
from wbml.data.eeg import load
from wbml.experiment import WorkingDirectory

from gpar import GPARRegressor

if __name__ == "__main__":
    wbml.out.report_time = True
    wd = WorkingDirectory("_experiments", "eeg")

    _, train, test = load()

    x = np.array(train.index)
    y = np.array(train)

    # Fit and predict GPAR.
    model = GPARRegressor(
        scale=0.02,
        linear=False,
        nonlinear=True,
        nonlinear_scale=1.0,
        noise=0.01,
        impute=True,
        replace=False,
        normalise_y=True,
    )
    model.fit(x, y)
    means, lowers, uppers = model.predict(
        x, num_samples=200, credible_bounds=True, latent=True
    )

    # Report SMSE.
    pred = pd.DataFrame(means, index=train.index, columns=train.columns)
    smse = wbml.metric.smse(pred, test)
    wbml.out.kv("SMSEs", smse.dropna())
    wbml.out.kv("Average SMSEs", smse.mean())

    # Name of output to plot.
    name = "F2"

    # Plot the result.
    plt.figure(figsize=(12, 1.75))
    wbml.plot.tex()

    p = list(train.columns).index(name)
    plt.plot(x, means[:, p], style="pred")
    plt.fill_between(x, lowers[:, p], uppers[:, p], style="pred")
    plt.scatter(x, y[:, p], style="train")
    plt.scatter(test[name].index, test[name], style="test")
    plt.xlabel("Time (second)")
    plt.xlim(0.4, 1)
    plt.ylabel(f"{name} (volt)")
    wbml.plot.tweak(legend=False)

    plt.tight_layout()
    plt.savefig(wd.file("eeg.pdf"))
