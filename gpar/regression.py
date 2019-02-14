# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import torch
from lab.torch import B
from stheno.torch import Graph, GP, EQ, Linear
from varz import Vars, minimise_l_bfgs_b

from .model import GPAR

__all__ = ['GPARRegressor']
log = logging.getLogger(__name__)


def _uprank(x):
    if np.ndim(x) > 2:
        raise ValueError('Invalid rank {}.'.format(np.ndims(x)))
    while 0 <= np.ndim(x) < 2:
        x = np.expand_dims(x, 1)
    return B.array(x)


def _model_generator(vs,
                     m,
                     p,
                     scale,
                     linear,
                     linear_slope,
                     nonlinear,
                     nonlinear_scale,
                     noise):
    def model():
        # Start out with a constant kernel.
        kernel = vs.bnd(name=(p, 'constant'), group=p, init=noise)

        # Add nonlinear kernel over inputs.
        scales = vs.bnd(name=(p, 'I/NL/scales'), group=p,
                        init=scale * B.ones(m))
        variance = vs.bnd(name=(p, 'I/NL/var'), group=p, init=1.)
        input_kernel = variance * EQ().stretch(scales)
        kernel += input_kernel.select(list(range(m))) if p > 1 else input_kernel

        # Add linear kernel if asked for.
        if linear:
            slopes = vs.bnd(name=(p, 'IO/L/slopes'), group=p,
                            init=linear_slope * B.ones(m + p - 1))
            kernel += Linear().stretch(1 / slopes)

        # Add nonlinear kernel over outputs if asked for.
        if nonlinear and p > 1:
            scales = vs.bnd(name=(p, 'O/NL/scales'), group=p,
                            init=nonlinear_scale * B.ones(p - 1))
            variance = vs.bnd(name=(p, 'O/NL/var'), group=p, init=1.)
            inds = list(range(m, m + p - 1))
            kernel += variance * EQ().stretch(scales).select(inds)

        # Return model and noise.
        return GP(kernel=kernel, graph=Graph()), \
               vs.bnd(name=(p, 'noise'), group=p, init=noise)

    return model


def _construct_gpar(reg, vs, m, p):
    # Check if inducing points are used.
    if reg.x_ind is not None:
        x_ind = vs.get(name='inducing_points', init=reg.x_ind)
    else:
        x_ind = None

    # Construct GPAR model layer by layer.
    gpar = GPAR(replace=reg.replace, impute=reg.impute, x_ind=x_ind)
    for i in range(1, p + 1):
        gpar = gpar.add_layer(_model_generator(vs, m, i, **reg.model_config))

    # Return GPAR model.
    return gpar


class GPARRegressor(object):
    """GPAR regressor.

    Args:
        replace (bool, optional): Replace observations with predictive means.
            Helps the model deal with noisy data points. Defaults to `True`.
        impute (bool, optional): Impute data with predictive means to make the
            data set closed downwards. Helps the model deal with missing data.
            Defaults to `True`.
        scale (float, optional): Initial value for the length scale over the
            inputs. Defaults to `1.0`.
        linear (bool, optional): Use linear dependencies between outputs.
            Defaults to `True`.
        linear_slope (float, optional): Initial value for the slope of the
            linear dependencies. Defaults to `0.1`.
        nonlinear (bool, optional): Use nonlinear dependencies between outputs.
            Defaults to `True`.
        nonlinear_scale (float, optional): Initial value to the length scale
            over the outputs. Defaults to `0.1`.
        noise (float, optional): Initial value for the observation noise.
            Defaults to `0.1`.
        x_ind (tensor, optional): Locations of inducing points. Set to `None`
            if inducing points should not be used. Defaults to `None`.

    Attributes:
        replace (bool): Replace observations with predictive means.
        impute (bool): Impute missing data with predictive means to make the
            data set closed downwards.
        sparse (bool): Use inducing points.
        x_ind (tensor): Locations of inducing points.
        model_config (dict): Summary of model configuration.
        vs (:class:`varz.Vars`): Model parameters.
        is_fit (bool): The model is fit.
        x (tensor): Inputs of training data.
        y (tensor): Outputs of training data.
        n (int): Number of training data points.
        m (int): Number of input features.
        p (int): Number of outputs.
    """

    def __init__(self,
                 replace=True,
                 impute=True,
                 scale=1.0,
                 linear=True,
                 linear_slope=0.1,
                 nonlinear=True,
                 nonlinear_scale=0.1,
                 noise=0.1,
                 x_ind=None):
        # Model configuration.
        self.replace = replace
        self.impute = impute
        self.sparse = x_ind is not None
        self.x_ind = None if x_ind is None else _uprank(x_ind)
        self.model_config = {
            'scale': scale,
            'linear': linear,
            'linear_slope': linear_slope,
            'nonlinear': nonlinear,
            'nonlinear_scale': nonlinear_scale,
            'noise': noise
        }

        # Model fitting.
        self.vs = Vars(dtype=torch.float64)
        self.is_fit = False
        self.x = None  # Inputs of training data
        self.y = None  # Outputs of training data
        self.n = None  # Number of data points
        self.m = None  # Number of input features
        self.p = None  # Number of outputs

    def fit(self, x, y, progressive=False, greedy=False):
        """Fit the model to data.

        Args:
            x (tensor): Inputs of training data.
            y (tensor): Outputs of training data.
            progressive (bool, optional): Train layer by layer instead of all
                layers at once. Defaults to `False`.
            greedy (bool, optional): Greedily determine the ordering of the
                outputs. Defaults to `False`.
        """
        if greedy:
            raise NotImplementedError('Greedy search is not implemented yet.')

        # Store data.
        self.x, self.y = _uprank(x), _uprank(y)
        self.n, self.m = self.x.shape
        self.p = self.y.shape[1]

        # Determine extra variables to optimise at every step.
        if self.sparse:
            names = ['inducing_points']
        else:
            names = []

        # Optimise layers by layer or all layers simultaneously.
        if progressive:
            # Check whether to only optimise the last layer.
            only_last = not (self.replace or self.impute or self.sparse)

            # Fit layer by layer.
            for i in range(1, self.p + 1):
                def objective(vs):
                    gpar = _construct_gpar(self, vs, self.m, i)
                    return -gpar.logpdf(self.x, self.y,
                                        only_last_layer=only_last)

                minimise_l_bfgs_b(objective,
                                  self.vs,
                                  names=names,
                                  groups=[i],
                                  trace=True)
        else:
            # Fit all layers simultaneously.
            def objective(vs):
                gpar = _construct_gpar(self, vs, self.m, self.p)
                return -gpar.logpdf(self.x, self.y, only_last_layer=False)

            minimise_l_bfgs_b(objective,
                              self.vs,
                              names=names,
                              groups=list(range(1, self.p + 1)),
                              trace=True)

        # Store that the model is fit.
        self.is_fit = True

    def sample(self, x, p=None, posterior=False, num_samples=1, latent=False):
        """Sample from the prior or posterior.

        Args:
            x (tensor): Inputs to sample at.
            p (int, optional): Number of outputs to sample if sampling from
                the prior.
            posterior (bool, optional): Sample from the prior instead of the
                posterior.
            num_samples (int, optional): Number of samples. Defaults to `1`.
            latent (bool, optional): Sample the latent function instead of
                observations. Defaults to `False`.

        Returns:
            list[tensor]: Prior samples. If only a single sample is
                generated, it will be returned directly instead of in a list.
        """
        x = _uprank(x)

        # Check that model is fit if sampling from the posterior.
        if posterior and not self.is_fit:
            raise RuntimeError('Must fit model before sampling form the '
                               'posterior.')
        # Check that the number of outputs is specified if sampling from the
        # prior.
        elif not posterior and p is None:
            raise ValueError('Must specify number of outputs to sample.')

        if posterior:
            # Construct posterior GPAR.
            gpar = _construct_gpar(self, self.vs, self.m, self.p)
            gpar = gpar | (self.x, self.y)
        else:
            # Construct prior GPAR.
            gpar = _construct_gpar(self, self.vs, B.shape_int(x)[1], p)

        # Sample and return.
        samples = [gpar.sample(x, latent=latent).detach().numpy()
                   for _ in range(num_samples)]
        return samples[0] if num_samples == 1 else samples

    def predict(self, x, num_samples=100, latent=True, credible_bounds=False):
        """Predict at new inputs.

        Args:
            x (tensor): Inputs to predict at.
            num_samples (int, optional): Number of samples. Defaults to `100`.
            latent (bool, optional): Predict the latent function instead of
                observations. Defaults to `True`.
            credible_bounds (bool, optional): Also return 95% central marginal
                credible bounds for the predictions.

        Returns:
            tensor: Predictive means. If `credible_bounds` is set to true,
                a three-tuple will be returned containing the predictive means,
                lower credible bounds, and upper credible bounds.
        """
        # Sample from posterior.
        samples = self.sample(
            x, num_samples=num_samples, latent=latent, posterior=True)

        # Compute mean.
        mean = np.mean(samples, axis=0)

        if credible_bounds:
            # Also return lower and upper credible bounds if asked for.
            lowers = np.percentile(samples, 2.5, axis=0)
            uppers = np.percentile(samples, 100 - 2.5, axis=0)
            return mean, lowers, uppers
        else:
            return mean
