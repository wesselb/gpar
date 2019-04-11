import pickle

import matplotlib.pyplot as plt
import numpy as np
from wbml.data.air_temp import load as load_temp

# Load data.
data = load_temp()

# Create lookups.
lookup_place = {'Chi/temp': 'Chimet', 'Cam/temp': 'Cambermet'}
lookup_size = {0: '10 Days', 1: '15 Days', 2: '1 Month'}

# Plot the results.
plt.figure(figsize=(15, 4))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

for d_size in [0, 1, 2]:
    d_all, d_train, d_tests = data[d_size]

    # Load predictions.
    with open('examples/paper/air_temp_results{}.pickle'.format(d_size),
              'rb') as f:
        preds = pickle.load(f)

    # Compute SMSEs for the first two data sets; the others are the extended
    # ones.
    smses = []
    for (mean, _, _), d in list(zip(preds, d_tests))[:2]:
        y_i = d_train.names.index(d.names[0])
        mse_mean = np.nanmean((d.y - np.nanmean(d.y,
                                                axis=0, keepdims=True)) ** 2)
        mse_gpar = np.nanmean((d.y - mean[:, y_i:y_i + 1]) ** 2)
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
        x = d_test.x.squeeze()
        x_ext = d_test_ext.x.squeeze()

        # Plot prediction.
        y_i = d_train.names.index(d_test.names[0])
        plt.plot(x_ext, mean[:, y_i], c='tab:blue')
        plt.fill_between(x_ext, lowers[:, y_i], uppers[:, y_i],
                         facecolor='tab:blue', alpha=.25)

        # Plot data.
        plt.scatter(x_ext, d_test_ext.y, c='tab:green', s=2)
        plt.scatter(x, d_test.y, c='tab:orange', s=2)

        # Finalise plot.
        plt.xlim(x_ext.min(), x_ext.max())
        plt.ylim(10, 30)
        plt.yticks([15, 20, 25])
        if i == 0:
            plt.title(lookup_size[d_size] + ' (SMSE: {:.3f})'.format(smse))
        if i == 1:
            plt.xlabel('Time (day)')
        if d_size == 0:
            plt.ylabel('{}\nTemp. (celsius)'
                       ''.format(lookup_place[d_test.names[0]]))

plt.tight_layout()
plt.savefig('examples/paper/air_temp_prediction.pdf')
plt.show()
