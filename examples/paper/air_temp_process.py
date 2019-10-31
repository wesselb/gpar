import pickle

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
from wbml.data.air_temp import load as load_temp


def date_to_day(dt):
    return dt.day + (dt.hour + (dt.minute + dt.second / 60) / 60) / 24


# Load data.
data = load_temp()

# Create lookups.
lookup_place = {('temp', 'Chi'): 'Chimet', ('temp', 'Cam'): 'Cambermet'}
lookup_size = {0: '10 Days', 1: '15 Days', 2: '1 Month'}

# Plot the results.
plt.figure(figsize=(15, 4))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

for d_size in [0, 1, 2]:
    d_all, d_train, d_tests = data[d_size]

    # Load predictions.
    with open(f'examples/paper/air_temp_results{d_size}.pickle', 'rb') as f:
        preds = pickle.load(f)

    # Compute SMSEs for the first two data sets; the others are the extended
    # ones.
    smses = []
    for (mean, _, _), d in list(zip(preds, d_tests))[:2]:
        y_i = list(d_train.columns).index(d.columns[0])
        mse_mean = \
            np.nanmean((d - np.nanmean(d, axis=0, keepdims=True)) ** 2)
        mse_gpar = np.nanmean((d - mean[:, y_i:y_i + 1]) ** 2)
        smses.append(mse_gpar / mse_mean)
    smse = np.mean(smses)

    # Construct plots.
    for i, (mean, lowers, uppers) in enumerate(preds[2:]):
        ax = plt.subplot(2, 3, d_size + i * 3 + 1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # Extract test set and extended test set.
        d_test, d_test_ext = d_tests[i], d_tests[i + 2]
        x = list(map(date_to_day, d_test.index))
        x_ext = list(map(date_to_day, d_test_ext.index))

        # Plot prediction.
        y_i = list(d_train.columns).index(d_test.columns[0])
        plt.plot(x_ext, mean[:, y_i], c='tab:blue')
        plt.fill_between(x_ext, lowers[:, y_i], uppers[:, y_i],
                         facecolor='tab:blue', alpha=.25)

        # Plot data.
        plt.scatter(x_ext, d_test_ext, c='tab:green', s=2)
        plt.scatter(x, d_test, c='tab:orange', s=2)

        # Finalise plot.
        plt.xlim(x_ext[0], x_ext[-1])
        plt.ylim(10, 30)
        plt.yticks([15, 20, 25])
        if i == 0:
            plt.title(lookup_size[d_size] + ' (SMSE: {:.3f})'.format(smse))
        if i == 1:
            plt.xlabel('Time (day)')
        if d_size == 0:
            plt.ylabel('{}\nTemp. (celsius)'
                       ''.format(lookup_place[d_test.columns[0]]))

plt.tight_layout()
plt.savefig('examples/paper/air_temp_prediction.pdf')
plt.show()
