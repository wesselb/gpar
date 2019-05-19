# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
import sys

import numpy as np
import torch
from lab.torch import B
from stheno.torch import Graph, GP, EQ, RQ, Delta, Linear, ZeroKernel
from varz import Vars
from varz.torch import minimise_l_bfgs_b

from .model import GPAR, per_output

__all__ = ['GPARRegressor', 'log_transform', 'squishing_transform']
log = logging.getLogger(__name__)

#: Log transform for the data.
log_transform = (B.log, B.exp)

#: Squishing transform for the data.
squishing_transform = (lambda x: B.sign(x) * B.log(1 + B.abs(x)),
                       lambda x: B.sign(x) * (B.exp(B.abs(x)) - 1))


def _vector_from_init(init, length):
    # If only a single value is given, create ones.
    if np.size(init) == 1:
        return init * np.ones(length)

    # Multiple values are given. Check that enough values are available.
    init_squeezed = np.squeeze(init)
    if np.ndim(init_squeezed) != 1:
        raise ValueError('Incorrect shape {} of hyperparameters.'
                         ''.format(np.shape(init)))
    if np.size(init_squeezed) < length:  # Squeezed doesn't change size.
        raise ValueError('Not enough hyperparameters specified.')

    # Return initialisation.
    return np.array(init_squeezed)[:length]


def _determine_indices(m, pi, markov):
    # Build in the Markov structure: juggle with the indices of the outputs.
    p_last = pi - 1  # Index of last output that is given as input.
    p_start = 0 if markov is None else max(p_last - (markov - 1), 0)
    p_num = p_last - p_start + 1

    # Determine the indices corresponding to the outputs and inputs.
    m_inds = list(range(m))
    p_inds = list(range(m + p_start, m + p_last + 1))

    return m_inds, p_inds, p_num


def _model_generator(vs,
                     m,  # This is the _number_ of inputs.
                     pi,  # This is the _index_ of the output modelled.
                     scale,
                     scale_tie,
                     per,
                     per_period,
                     per_scale,
                     per_decay,
                     input_linear,
                     input_linear_scale,
                     linear,
                     linear_scale,
                     nonlinear,
                     nonlinear_scale,
                     rq,
                     markov,
                     noise):
    def model():
        # Start with a zero kernels.
        kernel_inputs = ZeroKernel()  # Kernel over inputs.
        kernel_outputs = ZeroKernel()  # Kernel over outputs.

        # Determine indices corresponding to the inputs and outputs.
        m_inds, p_inds, p_num = _determine_indices(m, pi, markov)

        # Add nonlinear kernel over the inputs.
        variance = vs.bnd(name='{}/input/var'.format(pi), init=1.)
        scales = vs.bnd(name='{}/input/scales'.format(0 if scale_tie else pi),
                        init=_vector_from_init(scale, m))
        if rq:
            k = RQ(vs.bnd(name='{}/input/alpha', init=1e-2,
                          lower=1e-3, upper=1e3))
        else:
            k = EQ()
        kernel_inputs += variance * k.stretch(scales)

        # Add a locally periodic kernel over the inputs.
        if per:
            variance = vs.bnd(name='{}/input/per/var'.format(pi), init=1.)
            scales = vs.bnd(name='{}/input/per/scales'.format(pi),
                            init=_vector_from_init(per_scale, 2 * m))
            periods = vs.bnd(name='{}/input/per/pers'.format(pi),
                             init=_vector_from_init(per_period, m))
            decays = vs.bnd(name='{}/input/per/decay'.format(pi),
                            init=_vector_from_init(per_decay, m))
            kernel_inputs += variance * \
                             EQ().stretch(scales).periodic(periods) * \
                             EQ().stretch(decays)

        # Add a linear kernel over the inputs.
        if input_linear:
            scales = vs.bnd(name='{}/input/lin/scales'.format(pi),
                            init=_vector_from_init(input_linear_scale, m))
            const = vs.get(name='{}/input/lin/const'.format(pi), init=1.)
            kernel_inputs += Linear().stretch(scales) + const

        # Add linear kernel over the outputs.
        if linear and pi > 0:
            scales = vs.bnd(name='{}/output/lin/scales'.format(pi),
                            init=_vector_from_init(linear_scale, p_num))
            kernel_outputs += Linear().stretch(scales)

        # Add nonlinear kernel over the outputs.
        if nonlinear and pi > 0:
            variance = vs.bnd(name='{}/output/nonlin/var'.format(pi), init=1.)
            scales = vs.bnd(name='{}/output/nonlin/scales'.format(pi),
                            init=_vector_from_init(nonlinear_scale, p_num))
            if rq:
                k = RQ(vs.bnd(name='{}/output/nonlin/alpha'.format(pi),
                              init=1e-2, lower=1e-3, upper=1e3))
            else:
                k = EQ()
            kernel_outputs += variance * k.stretch(scales)

        # Construct noise kernel.
        variance = vs.bnd(name='{}/noise'.format(pi),
                          init=_vector_from_init(noise, pi + 1)[pi],
                          lower=1e-8)  # Allow noise to be small.
        kernel_noise = variance * Delta()

        # Construct model and return.
        graph = Graph()
        f = GP(kernel_inputs.select(m_inds) +
               kernel_outputs.select(p_inds), graph=graph)
        e = GP(kernel_noise, graph=graph)
        return f, e

    return model


def _construct_gpar(reg, vs, m, p):
    # Construct GPAR model layer by layer.
    gpar = GPAR(replace=reg.replace, impute=reg.impute, x_ind=reg.x_ind)
    for pi in range(p):
        gpar = gpar.add_layer(_model_generator(vs, m, pi, **reg.model_config))
    return gpar


class GPARRegressor(object):
    """GPAR regressor.

    Args:
        replace (bool, optional): Replace observations with predictive means.
            Helps the model deal with noisy data points. Defaults to `False`.
        impute (bool, optional): Impute data with predictive means to make the
            data set closed downwards. Helps the model deal with missing data.
            Defaults to `True`.
        scale (tensor, optional): Initial value(s) for the length scale(s) over
            the inputs. Defaults to `1.0`.
        scale_tie (bool, optional): Tie the length scale(s) over the inputs.
            Defaults to `False`.
        per (bool, optional): Use a locally periodic kernel over the inputs.
            Defaults to `False`.
        per_period (tensor, optional): Initial value(s) for the period(s) of the
            locally periodic kernel. Defaults to `1.0`.
        per_scale (tensor, optional): Initial value(s) for the length scale(s)
            of the locally periodic kernel. Defaults to `1.0`.
        per_decay (tensor, optional): Initial value(s) for the length scale(s)
            of the local change of the locally periodic kernel. Defaults to
            `10.0`.
        input_linear (bool, optional): Use a linear kernel over the inputs.
            Defaults to `False`.
        input_linear_scale (tensor, optional): Initial value(s) for the length
            scale(s) of the linear kernel over the inputs. Defaults to `100.0`.
        linear (bool, optional): Use linear dependencies between outputs.
            Defaults to `True`.
        linear_scale (tensor, optional): Initial value(s) for the length
            scale(s) of the linear dependencies. Defaults to `100.0`.
        nonlinear (bool, optional): Use nonlinear dependencies between outputs.
            Defaults to `True`.
        nonlinear_scale (tensor, optional): Initial value(s) for the length
            scale(s) over the outputs. Defaults to `0.1`.
        rq (bool, optional): Use rational quadratic (RQ) kernels instead of
            exponentiated quadratic (EQ) kernels. Defaults to `False`.
        markov (int, optional): Markov order of conditionals. Set to `None` to
            have a fully connected structure. Defaults to `None`.
        noise (tensor, optional): Initial value(s) for the observation noise(s).
            Defaults to `0.01`.
        x_ind (tensor, optional): Locations of inducing points. Set to `None`
            if inducing points should not be used. Defaults to `None`.
        normalise_y (bool, optional): Normalise outputs. Defaults to `True`.
        transform_y (tuple, optional): Tuple containing a transform and its
            inverse, which should be applied to the data before fitting.
            Defaults to the identity transform.

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
        normalise_y (bool): Normalise outputs.
    """

    def __init__(self,
                 replace=False,
                 impute=True,
                 scale=1.0,
                 scale_tie=False,
                 per=False,
                 per_period=1.0,
                 per_scale=1.0,
                 per_decay=10.0,
                 input_linear=False,
                 input_linear_scale=100.0,
                 linear=True,
                 linear_scale=100.0,
                 nonlinear=False,
                 nonlinear_scale=1.0,
                 rq=False,
                 markov=None,
                 noise=0.1,
                 x_ind=None,
                 normalise_y=True,
                 transform_y=(lambda x: x, lambda x: x)):
        # Model configuration.
        self.replace = replace
        self.impute = impute
        self.sparse = x_ind is not None
        if x_ind is None:
            self.x_ind = None
        else:
            self.x_ind = torch.tensor(B.uprank(x_ind))
        self.model_config = {
            'scale': scale,
            'scale_tie': scale_tie,
            'per': per,
            'per_period': per_period,
            'per_scale': per_scale,
            'per_decay': per_decay,
            'input_linear': input_linear,
            'input_linear_scale': input_linear_scale,
            'linear': linear,
            'linear_scale': linear_scale,
            'nonlinear': nonlinear,
            'nonlinear_scale': nonlinear_scale,
            'rq': rq,
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

        # Output normalisation and transformation.
        self.normalise_y = normalise_y
        self._unnormalise_y, self._normalise_y = lambda x: x, lambda x: x
        self._transform_y, self._untransform_y = transform_y

    def get_variables(self):
        """Construct a dictionary containing all the hyperparameters.

        Returns:
            dict: Dictionary mapping variable names to variable values.
        """
        variables = {}
        for name in self.vs.index_by_name.keys():
            variables[name] = self.vs[name].detach().numpy()
        return variables

    def fit(self,
            x,
            y,
            greedy=False,
            fix=True,
            **kw_args):
        """Fit the model to data.

        Further takes in keyword arguments for `Varz.minimise_l_bfgs_b`.

        Args:
            x (tensor): Inputs of training data.
            y (tensor): Outputs of training data.
            greedy (bool, optional): Greedily determine the ordering of the
                outputs. Defaults to `False`.
            fix (bool, optional): Fix the parameters of a layer after
                training it. If set to `False`, the likelihood are
                accumulated and all parameters are optimised at every step.
                Defaults to `True`.
        """
        if greedy:
            raise NotImplementedError('Greedy search is not implemented yet.')

        # Store data.
        self.x = torch.tensor(B.uprank(x))
        self.y = torch.tensor(self._transform_y(B.uprank(y)))
        self.n, self.m = self.x.shape
        self.p = self.y.shape[1]

        # Perform normalisation, carefully handling missing values.
        if self.normalise_y:
            means, stds = [], []
            for i in range(self.p):
                # Filter missing observations.
                available = ~B.isnan(self.y[:, i])
                y_i = self.y[available, i]

                # Calculate mean.
                means.append(B.mean(y_i))

                # Calculate std: safely handle the zero case.
                std = B.std(y_i)
                if std > 0:
                    stds.append(std)
                else:
                    stds.append(B.cast(B.dtype(std), 1))

            # Stack into a vector and create normalisers.
            means, stds = B.stack(*means)[None, :], B.stack(*stds)[None, :]

            def normalise_y(y_):
                return (y_ - means) / stds

            def unnormalise_y(y_):
                return y_ * stds + means

            # Save normalisers.
            self._normalise_y = normalise_y
            self._unnormalise_y = unnormalise_y

            # Perform normalisation.
            self.y = normalise_y(self.y)

        # Precompute the results of `per_output`. This can otherwise incur a
        # significant overhead if the number of outputs is large.
        y_cached = {k: list(per_output(self.y, keep=k)) for k in [True, False]}

        # Fit layer by layer.
        #   Note: `_construct_gpar` takes in the *number* of outputs.
        sys.stdout.write('Training conditionals (total: {}):'.format(self.p))
        sys.stdout.flush()
        for pi in range(self.p):
            sys.stdout.write(' {}'.format(pi + 1))
            sys.stdout.flush()

            # If we fix parameters of previous layers, we can precompute the
            # inputs. This speeds up the optimisation massively.
            if fix:
                gpar = _construct_gpar(self, self.vs, self.m, pi + 1)
                fixed_x, fixed_x_ind = gpar.logpdf(self.x, y_cached,
                                                   only_last_layer=True,
                                                   outputs=list(range(pi)),
                                                   return_inputs=True)

            def objective(vs):
                gpar = _construct_gpar(self, vs, self.m, pi + 1)
                # If the parameters of the previous layers are fixed, use the
                # precomputed inputs.
                if fix:
                    return -gpar.logpdf(fixed_x, y_cached,
                                        only_last_layer=True,
                                        outputs=[pi],
                                        x_ind=fixed_x_ind)
                else:
                    return -gpar.logpdf(self.x, y_cached, only_last_layer=False)

            # Determine names to optimise.
            if fix:
                names = ['{}/*'.format(pi)]
            else:
                names = ['{}/*'.format(i) for i in range(pi + 1)]

            # Perform the optimisation.
            minimise_l_bfgs_b(objective, self.vs, names=names, **kw_args)

        # Print newline to end progress bar.
        sys.stdout.write('\n')

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
        x = torch.tensor(B.uprank(x))
        y = torch.tensor(self._unnormalise_y(self._transform_y(B.uprank(y))))
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
        x = torch.tensor(B.uprank(x))

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
            gpar = _construct_gpar(self, self.vs, B.shape(x)[1], p)

        # Construct function to undo normalisation and transformation.
        def undo_transforms(y_):
            return self._untransform_y(self._unnormalise_y(y_))

        # Perform sampling.
        samples = []
        sys.stdout.write('Sampling (total: {}):'.format(num_samples))
        sys.stdout.flush()
        for i in range(num_samples):
            sys.stdout.write(' {}'.format(i + 1))
            sys.stdout.flush()
            samples.append(undo_transforms(gpar.sample(x, latent=latent))
                           .detach_().numpy())
        sys.stdout.write('\n')
        return samples[0] if num_samples == 1 else samples

    def predict(self, x, num_samples=100, latent=False, credible_bounds=False):
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
