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


def _vector_from_init(init, length):
    # If only a single value is given, create ones.
    if np.size(init) == 1:
        return init * np.ones(length)

    # Multiple values are given. Check that enough values are available.
    init = np.squeeze(init)
    if np.ndim(init) != 1:
        raise ValueError('Hyperparameters can be at most rank one.')
    if np.size(init) < length:
        raise ValueError('Not enough hyperparameters specified.')

    # Return initialisation.
    return np.array(init)[:length]


def _model_generator(vs,
                     m,  # This is the _number_ of inputs.
                     pi,  # This is the _index_ of the output modelled.
                     scale,
                     scale_tie,
                     linear,
                     linear_scale,
                     linear_with_inputs,
                     nonlinear,
                     nonlinear_scale,
                     nonlinear_with_inputs,
                     markov,
                     noise):
    def model():
        # Start with a zero kernel.
        kernel = 0

        # Build in the Markov structure: juggle with the indices of the outputs.
        p_last = pi - 1  # Index of last output that is given as input.
        p_start = 0 if markov is None else max(p_last - (markov - 1), 0)
        p_num = p_last - p_start + 1

        # Determine the indices corresponding to the outputs and inputs.
        m_inds = list(range(m))
        p_inds = list(range(m + p_start, m + p_last + 1))

        # Add nonlinear kernel over inputs.
        variance = vs.bnd(name=(pi, 'I/NL/var'), group=pi, init=1.)
        scales = vs.bnd(name=(0 if scale_tie else pi, 'I/NL/scales'),
                        group=0 if scale_tie else pi,
                        init=_vector_from_init(scale, m))
        kernel += variance * EQ().stretch(scales).select(m_inds)

        # Add linear kernel over outputs if asked for.
        if linear and pi > 0:
            scales = vs.bnd(name=(pi, 'O/L/scales'), group=pi,
                            init=_vector_from_init(linear_scale, p_num))
            kernel_linear = Linear().stretch(scales).select(p_inds)

            if linear_with_inputs:
                # Linear dependencies depend on the inputs.
                scales = vs.bnd(name=(pi, 'O/L/input_scales'), group=pi,
                                init=_vector_from_init(scale, m))
                kernel_linear *= EQ().stretch(scales).select(m_inds)

            # Add to the result.
            kernel += kernel_linear

        # Add nonlinear kernel over outputs if asked for.
        if nonlinear and pi > 0:
            # Nonlinear dependencies do not depend on the inputs.
            variance = vs.bnd(name=(pi, 'O/NL/var'), group=pi, init=1.)
            scales = vs.bnd(name=(pi, 'O/NL/scales'), group=pi,
                            init=_vector_from_init(nonlinear_scale, p_num))
            kernel_nonlinear = variance * EQ().stretch(scales).select(p_inds)

            if nonlinear_with_inputs:
                # Nonlinear dependencies depend on the inputs.
                scales = vs.bnd(name=(pi, 'O/NL/input_scales'), group=pi,
                                init=_vector_from_init(scale, m))
                kernel_nonlinear *= EQ().stretch(scales).select(m_inds)

            # Add to the result.
            kernel += kernel_nonlinear

        # Figure out initialisation for noise.
        if np.size(noise) > 1:
            if np.ndim(noise) == 1:
                noise_init = noise[pi]
            else:
                raise ValueError('Incorrect rank {} of noise.'
                                 ''.format(np.ndim(noise)))
        else:
            noise_init = noise

        # Return model and noise.
        return GP(kernel=kernel, graph=Graph()), \
               vs.bnd(name=(pi, 'noise'), group=pi, init=noise_init)

    return model


def _construct_gpar(reg, vs, m, p):
    # Check if inducing points are used.
    if reg.x_ind is not None:
        x_ind = vs.get(name='inducing_points', init=reg.x_ind)
    else:
        x_ind = None

    # Construct GPAR model layer by layer.
    gpar = GPAR(replace=reg.replace, impute=reg.impute, x_ind=x_ind)
    for pi in range(p):
        gpar = gpar.add_layer(_model_generator(vs, m, pi, **reg.model_config))

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
        scale (tensor, optional): Initial value(s) for the length scale(s) over
            the inputs. Defaults to `1.0`.
        scale_tie (bool, optional): Tie the length scale(s) over the inputs.
            Defaults to `False`.
        linear (bool, optional): Use linear dependencies between outputs.
            Defaults to `True`.
        linear_scale (tensor, optional): Initial value(s) for the scale(s) of
            the linear dependencies. Defaults to `100.0`.
        linear_with_inputs (bool, optional): Make the linear dependencies
            between the outputs dependent on the inputs. Defaults to `False`.
        nonlinear (bool, optional): Use nonlinear dependencies between outputs.
            Defaults to `True`.
        nonlinear_scale (tensor, optional): Initial value(s) for the length
            scale(s) over the outputs. Defaults to `0.1`.
        nonlinear_with_inputs (bool, optional): Make the nonlinear dependencies
            between the outputs dependent on the inputs. Defaults to `False`.
        markov (int, optional): Markov order of conditionals. Set to `None` to
            have a fully connected structure. Defaults to `None`.
        noise (tensor, optional): Initial value(s) for the observation noise(s).
            Defaults to `0.01`.
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
                 scale_tie=False,
                 linear=True,
                 linear_scale=100.0,
                 linear_with_inputs=False,
                 nonlinear=False,
                 nonlinear_scale=0.1,
                 nonlinear_with_inputs=False,
                 markov=None,
                 noise=0.01,
                 x_ind=None):
        # Model configuration.
        self.replace = replace
        self.impute = impute
        self.sparse = x_ind is not None
        self.x_ind = None if x_ind is None else _uprank(x_ind)
        self.model_config = {
            'scale': scale,
            'scale_tie': scale_tie,
            'linear': linear,
            'linear_scale': linear_scale,
            'linear_with_inputs': linear_with_inputs,
            'nonlinear': nonlinear,
            'nonlinear_scale': nonlinear_scale,
            'nonlinear_with_inputs': nonlinear_with_inputs,
            'markov': markov,
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

    def get_variables(self):
        """Construct a dictionary containing all the hyperparameters.

        Returns:
            dict: Dictionary mapping variable names to variable values.
        """
        variables = {}
        for name in self.vs.names.keys():
            variables[name] = self.vs[name].detach().numpy()
        return variables

    def fit(self,
            x,
            y,
            progressive=True,
            greedy=False,
            **kw_args):
        """Fit the model to data.

        Further takes in keyword arguments for `Varz.minimise_l_bfgs_b`.

        Args:
            x (tensor): Inputs of training data.
            y (tensor): Outputs of training data.
            progressive (bool, optional): Train layer by layer instead of all
                layers at once. Defaults to `True`.
            greedy (bool, optional): Greedily determine the ordering of the
                outputs. Defaults to `False`.
        """
        if greedy:
            raise NotImplementedError('Greedy search is not implemented yet.')

        # Store data.
        self.x, self.y = _uprank(x), _uprank(y)
        self.n, self.m = self.x.shape
        self.p = self.y.shape[1]

        # Optimise layers by layer or all layers simultaneously.
        if progressive:
            # Check whether to only optimise the last layer.
            only_last = not self.sparse

            # Fit layer by layer.
            for pi in range(self.p):
                def objective(vs):
                    # `_construct_gpar` takes in the _number_ of outputs.
                    gpar = _construct_gpar(self, vs, self.m, pi + 1)
                    return -gpar.logpdf(self.x, self.y,
                                        only_last_layer=only_last)

                # Determine extra variables to optimise.
                if self.sparse and pi == 0:
                    names = ['inducing_points']
                else:
                    names = []

                minimise_l_bfgs_b(objective,
                                  self.vs,
                                  names=names,
                                  groups=[pi],
                                  **kw_args)
        else:
            # Determine extra variables to optimise.
            names = ['inducing_points'] if self.sparse else []

            # Fit all layers simultaneously.
            def objective(vs):
                gpar = _construct_gpar(self, vs, self.m, self.p)
                return -gpar.logpdf(self.x, self.y, only_last_layer=False)

            minimise_l_bfgs_b(objective,
                              self.vs,
                              names=names,
                              groups=list(range(self.p)),
                              **kw_args)

        # Store that the model is fit.
        self.is_fit = True

    def logpdf(self, x, y, sample_missing=False, posterior=False):
        """Compute the logpdf of observations.

        Args:
            x (tensor): Inputs.
            y (tensor): Outputs.
            sample_missing (bool, optional): Sample missing data to compute an
                unbiased estimate of the pdf, *not* logpdf. Defaults to `False`.
            posterior (bool, optional): Compute logpdf under the posterior
                instead of the prior. Defaults to `False`.

        Returns
            float: Estimate of the logpdf.
        """
        x, y = _uprank(x), _uprank(y)
        m, p = x.shape[1], y.shape[1]

        if posterior and not self.is_fit:
            raise RuntimeError('Must fit model before computing the logpdf '
                               'under the posterior.')

        # Construct GPAR and sample logpdf.
        gpar = _construct_gpar(self, self.vs, m, p)
        if posterior:
            gpar = gpar | (self.x, self.y)
        return gpar.logpdf(x, y,
                           only_last_layer=False,
                           sample_missing=sample_missing).detach_().numpy()

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
        samples = self.sample(x,
                              num_samples=num_samples,
                              latent=latent,
                              posterior=True)

        # Compute mean.
        mean = np.mean(samples, axis=0)

        if credible_bounds:
            # Also return lower and upper credible bounds if asked for.
            lowers = np.percentile(samples, 2.5, axis=0)
            uppers = np.percentile(samples, 100 - 2.5, axis=0)
            return mean, lowers, uppers
        else:
            return mean
