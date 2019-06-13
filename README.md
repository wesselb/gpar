# [GPAR](http://github.com/wesselb/gpar)

[![Build](https://travis-ci.org/wesselb/gpar.svg?branch=master)](https://travis-ci.org/wesselb/gpar)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/gpar/badge.svg?branch=master&service=github)](https://coveralls.io/github/wesselb/gpar?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/gpar)

Implementation of the Gaussian Process Autoregressive Regression Model

See the [paper](https://arxiv.org/abs/1802.07182), and see the [docs](https://wesselb.github.io/gpar).

* [Installation](#installation)
* [Basic Usage](#basic-usage)
* [Features](#features)
    - [Input and Output Dependencies](#input-and-output-dependencies)
    - [Output Transformation](#output-transformation)
    - [Sampling](#sampling)
    - [Logpdf Computation](#logpdf-computation)
    - [Inducing Points](#inducing-points)
 * [Example (examples/paper/synthetic.py)](#example-examples-paper-synthetic-py)

## Installation
Before installing the package, please ensure that `gcc` and `gfortran` are 
available.
On OS X, these are both installed with `brew install gcc`;
users of Anaconda may want to instead consider `conda install gcc`.
Then simply

```
pip install gpar
```

## Basic Usage
A simple instance of GPAR can be created as follows:

```python
from gpar import GPARRegressor

gpar = GPARRegressor(replace=True, impute=True,
                     scale=1.0,
                     linear=True, linear_scale=100.0,
                     nonlinear=True, nonlinear_scale=1.0,
                     noise=0.1,
                     normalise_y=True)
```

Here the keyword arguments have the following meaning:

* `replace=True`: Replace data points with the posterior mean of the previous
    layer before feeding them into the next layer. This helps the model deal 
    with noisy data, but may discard important structure in the data if  the 
    fit is bad.
    
* `impute=True`: GPAR requires that data is _closed downwards_. If this is 
    not the case, the model will be unable to use part of the data. Setting 
    `impute` to `True` lets GPAR impute data points to ensure that the data is
    closed downwards.
    
* `scale=1.0`: Initialisation of the length scale with respect to the inputs.

* `linear=True`: Use linear dependencies between outputs.

* `linear_scale=True`: Initialisation of the length scale of the linear 
    dependencies between outputs.
    
* `nonlinear=True`: Also use nonlinear dependencies between outputs.

* `nonlinear_scale=1.0`: Initialisation of the length scale of the nonlinear 
    dependencies between outputs. _Important:_ this length scale applies 
    _after_ possible normalisation of the outputs (see below), in which case 
    `nonlinear_scale=1.0` corresponds to a simple, but nonlinear relationship.
    
* `noise=0.1`: Initialisation of the variance of the observation noise.

* `normalise_y=True`: Internally, work with a normalised version of the 
    outputs by subtracting the mean and dividing by the standard deviation.
    Predictions will be transformed back appropriately.

In the above, any scalar hyperparameter may be replaced by a list of values 
to initialise each layer separately, e.g. `scale=[1.0, 2.0]`. See the 
documentation for a full overview of the keywords that may be passed to 
`GPARRegressor`.

To fit GPAR, call `gpar.fit(x_train, y_train)` where `x_train` are the training 
inputs and `y_train` the training outputs. The inputs `x_train` must have shape 
$n$ or $n \times m$, where $n$ is  the number of data points and $m$ the 
number of input features, and the outputs `y_train` must have shape $n$ or $n 
\times p$, where $p$ is the number of outputs.

Finally, to make predictions, call

```python
means = gpar.predict(x_test, num_samples=100)
```

to get the predictive means, or 

```python
means, lowers, uppers = gpar.predict(x_test,
                                     num_samples=100, 
                                     credible_bounds=True)
```

to also get lower and upper 95% central marginal credible bounds. If you wish
 to predict the underlying latent function instead of the observed values, set
`latent=True` in the call to `GPARRegressor.predict`.

## Features

### Input and Output Dependencies
Using keywords arguments, `GPARRegressor` can be configured to specify the 
dependencies with respect to the inputs and between the outputs. The following
dependencies can be specified:

* **Linear input dependencies:** Set `linear_input=True` and specify the 
    length scale with `linear_input_scale`.
    
* **Nonlinear input dependencies:** This is enabled by default. The length 
    scale can be specified using `scale`. To tie these length scales across all
    layers, set `scale_tie=True`.
    
* **Locally periodic input dependencies:** Set `per=True` and specify the period
    with `per_period`, the length scale with `per_scale`, and the length 
    scale on which the periodicity changes with `per_decay`.
    
* **Linear output dependencies:** Set `linear=True` and specify the length 
    scale with `linear_scale`.
    
* **Nonlinear output dependencies:** Set `nonlinear=True` and specify the 
    length scale with `nonlinear_scale`.
    
All nonlinear kernels are exponentiated quadratic kernels. If you wish to 
instead use rational quadratic kernels, set `rq=True`.

All parameters can be set to a list of values to initialise the value for 
each layer separately.

To let every layer depend only the `k`th previous layers, set `markov=k`.


### Output Transformation

One may want to apply a transformation to the data before fitting the model, 
e.g. $y\mapsto\log(y)$ in the case of positive data. Such a transformation can
be specified by setting the `transform_y` keyword argument for `GPARRegressor`.
The following transformations are available:

* `log_transform`: $y \mapsto \log(y)$.

* `squishing_transform`: $y \mapsto \operatorname{sign}(y) \log(1 + |y|)$.


### Sampling

Sampling from the model can be done with `GPARRegressor.sample`. The keyword 
argument `num_samples` specifies the number of samples, and `latent` 
specifies whether to sample from the underlying latent function or the 
observed values. Sampling from the _prior_ and _posterior_ (model must be fit
first) can be done as follows:
 
```python
sample = gpar.sample(x, p=2)  # Sample two outputs from the prior.

sample = gpar.sample(x, posterior=True)  # Sample from the posterior.
```

### Logpdf Computation
The logpdf of data can be computed with `GPARRegressor.logpdf`. To compute the
logpdf under the posterior, set `posterior=True`. To sample missing data to 
compute an unbiased estimate of the *pdf*, *not logpdf*, set 
`sample_missing=True`.


### Inducing Points
Inducing points can be used to scale GPAR to large data sets. Simply set `x_ind`
to the locations of the inducing points in `GPARRegressor`.


## Example (`examples/paper/synthetic.py`)

![Prediction](https://raw.githubusercontent.com/wesselb/gpar/master/readme_example_prediction.png)

```python
import matplotlib.pyplot as plt
import numpy as np
from gpar.regression import GPARRegressor

# Create toy data set.
n = 200
x = np.linspace(0, 1, n)
noise = 0.1

# Draw functions depending on each other in complicated ways.
f1 = -np.sin(10 * np.pi * (x + 1)) / (2 * x + 1) - x ** 4
f2 = np.cos(f1) ** 2 + np.sin(3 * x)
f3 = f2 * f1 ** 2 + 3 * x
f = np.stack((f1, f2, f3), axis=0).T

# Add noise and subsample.
y = f + noise * np.random.randn(n, 3)
x_obs, y_obs = x[::8], y[::8]

# Fit and predict GPAR.
model = GPARRegressor(scale=0.1,
                      linear=True, linear_scale=10.,
                      nonlinear=True, nonlinear_scale=0.1,
                      noise=0.1,
                      impute=True, replace=True, normalise_y=False)
model.fit(x_obs, y_obs)
means, lowers, uppers = \
    model.predict(x, num_samples=200, credible_bounds=True, latent=True)

# Fit and predict independent GPs: set markov=0.
igp = GPARRegressor(scale=0.1,
                    linear=True, linear_scale=10.,
                    nonlinear=True, nonlinear_scale=0.1,
                    noise=0.1, markov=0, normalise_y=False)
igp.fit(x_obs, y_obs)
igp_means, igp_lowers, igp_uppers = \
    igp.predict(x, num_samples=200, credible_bounds=True, latent=True)

# Plot the result.
plt.figure(figsize=(12, 2.5))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

for i in range(3):
    ax = plt.subplot(1, 3, i + 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.scatter(x_obs, y_obs[:, i], label='Observations', c='black', s=15)
    plt.plot(x, f[:, i], label='Truth', c='tab:orange')
    plt.plot(x, means[:, i], label='GPAR', c='tab:blue')
    plt.fill_between(x, lowers[:, i], uppers[:, i],
                     facecolor='tab:blue', alpha=.25)
    plt.plot(x, igp_means[:, i], label='IGP', c='tab:green')
    plt.fill_between(x, igp_lowers[:, i], igp_uppers[:, i],
                     facecolor='tab:green', alpha=.25)
    plt.xlabel('$t$')
    plt.ylabel('$y_{}$'.format(i + 1))
    if i == 2:
        leg = plt.legend(facecolor='#eeeeee')
        leg.get_frame().set_linewidth(0)

plt.tight_layout()
plt.savefig('examples/paper/synthetic_prediction.pdf')
plt.show()
```




