import matplotlib.pyplot as plt
import numpy as np
from wbml.data.air_temp import load
from wbml.experiment import WorkingDirectory
import wbml.plot
import wbml.metric
import pandas as pd


def date_to_day(dt):
    return dt.day + (dt.hour + (dt.minute + dt.second / 60) / 60) / 24


wd = WorkingDirectory("_experiments", "air_temp", log="log_process.txt", observe=True)

# Load data.
data = load()

# Create lookups.
lookup_place = {("temp", "Chi"): "Chimet", ("temp", "Cam"): "Cambermet"}
lookup_size = {0: "10 Days", 1: "15 Days", 2: "1 Month"}

# Plot the results.
plt.figure(figsize=(15, 4))

for d_size in [0, 1, 2]:
    d_all, d_train, d_tests = data[d_size]

    # Load predictions.
    preds = wd.load(f"results{d_size}.pickle")

    # Compute SMSEs for the first two data sets; the others are the extended
    # ones.
    smses = []
    for (mean, _, _), d_test in list(zip(preds, d_tests))[:2]:
        mean = pd.DataFrame(mean, index=d_test.index, columns=d_train.columns)
        smse = wbml.metric.smse(mean, d_test).mean()
        smses.append(smse)
    smse = np.mean(smses)

    # Construct plots.
    for i, (mean, lowers, uppers) in enumerate(preds[2:]):
        plt.subplot(2, 3, d_size + i * 3 + 1)

        # Extract test set and extended test set.
        d_test, d_test_ext = d_tests[i], d_tests[i + 2]
        x = list(map(date_to_day, d_test.index))
        x_ext = list(map(date_to_day, d_test_ext.index))

        # Plot prediction.
        y_i = list(d_train.columns).index(d_test.columns[0])
        plt.plot(x_ext, mean[:, y_i], style="pred")
        plt.fill_between(x_ext, lowers[:, y_i], uppers[:, y_i], style="pred")

        # Plot data.
        plt.scatter(x_ext, d_test_ext, style="train", s=4)
        plt.scatter(x, d_test, style="test", s=4)

        # Finalise plot.
        plt.xlim(x_ext[0], x_ext[-1])
        plt.ylim(10, 30)
        plt.yticks([15, 20, 25])
        if i == 0:
            plt.title(lookup_size[d_size] + f" (SMSE: {smse:.3f})")
        if i == 1:
            plt.xlabel("Time (day)")
        if d_size == 0:
            plt.ylabel(f"{lookup_place[d_test.columns[0]]}\nTemp. (celsius)")

        wbml.plot.tweak(legend=False)

plt.tight_layout()
plt.savefig(wd.file("air_temp.pdf"))
