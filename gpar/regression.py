import logging

import numpy as np
import torch
from lab.torch import B
from matrix import AbstractMatrix
from plum import Dispatcher, Union
from stheno.torch import Measure, GP, EQ, RQ, Delta, Linear, ZeroKernel
from varz import Vars
from varz.torch import minimise_l_bfgs_b
from wbml.out import Counter

from .model import GPAR, per_output

__all__ = ["GPARRegressor", "log_transform", "squishing_transform"]

log = logging.getLogger(__name__)

_dispatch = Dispatcher()

#: Log transform for the data.
log_transform = (B.log, B.exp)

#: Squishing transform for the data.
squishing_transform = (
    lambda x: B.sign(x) * B.log(B.add(1, B.abs(x))),
    lambda x: B.sign(x) * B.subtract(B.exp(B.abs(x)), 1),
)


def _vector_from_init(init, length):
    # If only a single value is given, create ones.
    if np.size(init) == 1:
        return init * np.ones(length)

    # Multiple values are given. Check that enough values are available.
    init_squeezed = np.squeeze(init)
    if np.ndim(init_squeezed) != 1:
        raise ValueError(
            "Incorrect shape {} of hyperparameters." "".format(np.shape(init))
        )
    if np.size(init_squeezed) < length:  # Squeezed doesn't change size.
        raise ValueError("Not enough hyperparameters specified.")

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


@_dispatch
def _to_torch(x: B.NP):
    return torch.tensor(x)


@_dispatch
def _to_torch(x: Union[B.Torch, AbstractMatrix, type(None)]):
    return x


def _model_generator(
    vs,
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
    noise,
):
    def model():
        # Start with a zero kernels.
        kernel_inputs = ZeroKernel()  # Kernel over inputs.
        kernel_outputs = ZeroKernel()  # Kernel over outputs.

        # Determine indices corresponding to the inputs and outputs.
        m_inds, p_inds, p_num = _determine_indices(m, pi, markov)

        # Add nonlinear kernel over the inputs.
        variance = vs.bnd(name=f"{pi}/input/var", init=1.0)
        scales = vs.bnd(
            name=f"{0 if scale_tie else pi}/input/scales",
            init=_vector_from_init(scale, m),
        )
        if rq:
            k = RQ(vs.bnd(name=f"{pi}/input/alpha", init=1e-2, lower=1e-3, upper=1e3))
        else:
            k = EQ()
        kernel_inputs += variance * k.stretch(scales)

        # Add a locally periodic kernel over the inputs.
        if per:
            variance = vs.bnd(name=f"{pi}/input/per/var", init=1.0)
            scales = vs.bnd(
                name=f"{pi}/input/per/scales",
                init=_vector_from_init(per_scale, 2 * m),
            )
            periods = vs.bnd(
                name=f"{pi}/input/per/pers",
                init=_vector_from_init(per_period, m),
            )
            decays = vs.bnd(
                name=f"{pi}/input/per/decay",
                init=_vector_from_init(per_decay, m),
            )
            kernel_inputs += (
                variance * EQ().stretch(scales).periodic(periods) * EQ().stretch(decays)
            )

        # Add a linear kernel over the inputs.
        if input_linear:
            scales = vs.bnd(
                name=f"{pi}/input/lin/scales",
                init=_vector_from_init(input_linear_scale, m),
            )
            const = vs.get(name=f"{pi}/input/lin/const", init=1.0)
            kernel_inputs += Linear().stretch(scales) + const

        # Add linear kernel over the outputs.
        if linear and pi > 0:
            scales = vs.bnd(
                name=f"{pi}/output/lin/scales",
                init=_vector_from_init(linear_scale, p_num),
            )
            kernel_outputs += Linear().stretch(scales)

        # Add nonlinear kernel over the outputs.
        if nonlinear and pi > 0:
            variance = vs.bnd(name=f"{pi}/output/nonlin/var", init=1.0)
            scales = vs.bnd(
                name=f"{pi}/output/nonlin/scales",
                init=_vector_from_init(nonlinear_scale, p_num),
            )
            if rq:
                k = RQ(
                    vs.bnd(
                        name=f"{pi}/output/nonlin/alpha",
                        init=1e-2,
                        lower=1e-3,
                        upper=1e3,
                    )
                )
            else:
                k = EQ()
            kernel_outputs += variance * k.stretch(scales)

        # Construct noise kernel.
        variance = vs.bnd(
            name=f"{pi}/noise",
            init=_vector_from_init(noise, pi + 1)[pi],
            lower=1e-8,
        )  # Allow noise to be small.
        kernel_noise = variance * Delta()

        # Construct model and return.
        prior = Measure()
        f = GP(
            kernel_inputs.select(m_inds) + kernel_outputs.select(p_inds), measure=prior
        )
        e = GP(kernel_noise, measure=prior)
        return f, e

    return model


def _construct_gpar(reg, vs, m, p):
    # Construct GPAR model layer by layer.
    gpar = GPAR(replace=reg.replace, impute=reg.impute, x_ind=reg.x_ind)
    for pi in range(p):
        gpar = gpar.add_layer(_model_generator(vs, m, pi, **reg.model_config))
    return gpar


def _init_weights(w, y):
    if w is None:
        return B.ones(torch.float64, *B.shape(y))
    else:
        return B.uprank(_to_torch(w))


class GPARRegressor:
    """GPAR regressor.

    Args:
        replace (:obj:`bool`, optional): Replace observations with predictive means.
            Helps the model deal with noisy data points. Defaults to `False`.
        impute (:obj:`bool`, optional): Impute data with predictive means to make the
            data set closed downwards. Helps the model deal with missing data.
            Defaults to `True`.
        scale (tensor, optional): Initial value(s) for the length scale(s) over the
            inputs. Defaults to `1.0`.
        scale_tie (:obj:`bool`, optional): Tie the length scale(s) over the inputs.
            Defaults to `False`.
        per (:obj:`bool`, optional): Use a locally periodic kernel over the inputs.
            Defaults to `False`.
        per_period (tensor, optional): Initial value(s) for the period(s) of the
            locally periodic kernel. Defaults to `1.0`.
        per_scale (tensor, optional): Initial value(s) for the length scale(s) of the
            locally periodic kernel. Defaults to `1.0`.
        per_decay (tensor, optional): Initial value(s) for the length scale(s) of the
            local change of the locally periodic kernel. Defaults to `10.0`.
        input_linear (:obj:`bool`, optional): Use a linear kernel over the inputs.
            Defaults to `False`.
        input_linear_scale (tensor, optional): Initial value(s) for the length
            scale(s) of the linear kernel over the inputs. Defaults to `100.0`.
        linear (:obj:`bool`, optional): Use linear dependencies between outputs.
            Defaults to `True`.
        linear_scale (tensor, optional): Initial value(s) for the length scale(s) of
            the linear dependencies. Defaults to `100.0`.
        nonlinear (:obj:`bool`, optional): Use nonlinear dependencies between outputs.
            Defaults to `True`.
        nonlinear_scale (tensor, optional): Initial value(s) for the length scale(s)
            over the outputs. Defaults to `0.1`.
        rq (:obj:`bool`, optional): Use rational quadratic (RQ) kernels instead of
            exponentiated quadratic (EQ) kernels. Defaults to `False`.
        markov (:obj:`int`, optional): Markov order of conditionals. Set to `None` to
            have a fully connected structure. Defaults to `None`.
        noise (tensor, optional): Initial value(s) for the observation noise(s).
            Defaults to `0.01`.
        x_ind (tensor, optional): Locations of inducing points. Set to `None` if
            inducing points should not be used. Defaults to `None`.
        normalise_y (:obj:`bool`, optional): Normalise outputs. Defaults to `True`.
        transform_y (:obj:`tuple`, optional): Tuple containing a transform and its
            inverse, which should be applied to the data before fitting. Defaults to
            the identity transform.

    Attributes:
        replace (bool): Replace observations with predictive means.
        impute (bool): Impute missing data with predictive means to make the data set closed downwards.
        sparse (bool): Use inducing points.
        x_ind (tensor): Locations of inducing points.
        model_config (dict): Summary of model configuration.
        vs (:class:`varz.Vars`): Model parameters.
        is_conditioned (bool): The model is conditioned.
        x (tensor): Inputs of training data.
        y (tensor): Outputs of training data.
        w (vector): Weights for every time stamp.
        n (int): Number of training data points.
        m (int): Number of input features.
        p (int): Number of outputs.
        normalise_y (bool): Normalise outputs.
    """

    def __init__(
        self,
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
        transform_y=(lambda x: x, lambda x: x),
    ):
        # Model configuration.
        self.replace = replace
        self.impute = impute
        self.sparse = x_ind is not None
        if x_ind is None:
            self.x_ind = None
        else:
            self.x_ind = B.uprank(_to_torch(x_ind))
        self.model_config = {
            "scale": scale,
            "scale_tie": scale_tie,
            "per": per,
            "per_period": per_period,
            "per_scale": per_scale,
            "per_decay": per_decay,
            "input_linear": input_linear,
            "input_linear_scale": input_linear_scale,
            "linear": linear,
            "linear_scale": linear_scale,
            "nonlinear": nonlinear,
            "nonlinear_scale": nonlinear_scale,
            "rq": rq,
            "markov": markov,
            "noise": noise,
        }

        # Model fitting.
        self.vs = Vars(dtype=torch.float64)
        self.is_conditioned = False
        self.x = None  # Inputs of training data
        self.y = None  # Outputs of training data
        self.w = None  # Weights for every time stamp
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
        for name in self.vs.names:
            variables[name] = self.vs[name].detach().numpy()
        return variables

    def condition(self, x, y, w=None):
        """Condition the model on data, without training.

        Args:
            x (tensor): Inputs of training data.
            y (tensor): Outputs of training data.
            w (tensor, optional): Weights of training data.
        """
        # Store data.
        self.x = B.uprank(_to_torch(x))
        self.y = self._transform_y(B.uprank(_to_torch(y)))
        self.w = _init_weights(w, self.y)
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
                return B.divide(B.subtract(y_, means), stds)

            def unnormalise_y(y_):
                return B.add(B.multiply(y_, stds), means)

            # Save normalisers.
            self._normalise_y = normalise_y
            self._unnormalise_y = unnormalise_y

            # Perform normalisation.
            self.y = normalise_y(self.y)

        # Store that the model is conditioned.
        self.is_conditioned = True

    def fit(self, x, y, w=None, greedy=False, fix=True, **kw_args):
        """Fit the model to data.

        Further takes in keyword arguments for `Varz.minimise_l_bfgs_b`.

        Args:
            x (tensor): Inputs of training data.
            y (tensor): Outputs of training data.
            w (tensor, optional): Weights of training data.
            greedy (bool, optional): Greedily determine the ordering of the outputs.
                Defaults to `False`.
            fix (bool, optional): Fix the parameters of a layer after training it. If
                set to `False`, the likelihood are accumulated and all parameters are
                optimised at every step. Defaults to `True`.
        """
        # Conditioned the model before fitting.
        self.condition(x, y, w)

        if greedy:
            raise NotImplementedError("Greedy search is not implemented yet.")

        # Precompute the results of `per_output`. This can otherwise incur a
        # significant overhead if the number of outputs is large.
        y_cached = {k: list(per_output(self.y, self.w, keep=k)) for k in [True, False]}

        # Fit layer by layer. NOTE: `_construct_gpar` takes in the *number* of outputs.
        with Counter(name="Training conditionals", total=self.p) as counter:
            for pi in range(self.p):
                counter.count()

                # If we fix parameters of previous layers, we can precompute the
                # inputs. This speeds up the optimisation massively.
                if fix:
                    gpar = _construct_gpar(self, self.vs, self.m, pi + 1)
                    fixed_x, fixed_x_ind = gpar.logpdf(
                        self.x,
                        y_cached,
                        None,
                        only_last_layer=True,
                        outputs=list(range(pi)),
                        return_inputs=True,
                    )

                def objective(vs):
                    gpar = _construct_gpar(self, vs, self.m, pi + 1)
                    # If the parameters of the previous layers are fixed, use
                    # the precomputed inputs.
                    if fix:
                        return -gpar.logpdf(
                            fixed_x,
                            y_cached,
                            None,
                            only_last_layer=True,
                            outputs=[pi],
                            x_ind=fixed_x_ind,
                        )
                    else:
                        return -gpar.logpdf(
                            self.x, y_cached, None, only_last_layer=False
                        )

                # Determine names to optimise.
                if fix:
                    names = [f"{pi}/*"]
                else:
                    names = [f"{i}/*" for i in range(pi + 1)]

                # Perform the optimisation.
                minimise_l_bfgs_b(objective, self.vs, names=names, **kw_args)

    def logpdf(self, x, y, w=None, sample_missing=False, posterior=False):
        """Compute the logpdf of observations.

        If either `x` or `y` is a PyTorch tensor, then the result will not be
        detached from the computation graph and converted to NumPy.

        Args:
            x (tensor): Inputs.
            y (tensor): Outputs.
            w (tensor, optional): Weights.
            sample_missing (bool, optional): Sample missing data to compute an
                unbiased estimate of the pdf, *not* logpdf. Defaults to `False`.
            posterior (bool, optional): Compute logpdf under the posterior
                instead of the prior. Defaults to `False`.

        Returns
            float: Estimate of the logpdf.
        """
        # Check whether either `x` or `y` was already PyTorch.
        any_torch = isinstance(x, B.Torch) or isinstance(y, B.Torch)

        x = B.uprank(_to_torch(x))
        y = self._unnormalise_y(self._transform_y(B.uprank(_to_torch(y))))
        w = _init_weights(w, y)
        m, p = x.shape[1], y.shape[1]

        if posterior and not self.is_conditioned:
            raise RuntimeError(
                "Must condition or fit model before computing "
                "the logpdf under the posterior."
            )

        # Construct GPAR and compute logpdf.
        gpar = _construct_gpar(self, self.vs, m, p)
        if posterior:
            gpar = gpar | (self.x, self.y, self.w)
        value = gpar.logpdf(
            x, y, w, only_last_layer=False, sample_missing=sample_missing
        )

        # If neither `x` nor `y` was already a PyTorch tensor, return the
        # result as NumPy.
        if not any_torch:
            value = value.detach_().numpy()

        return value

    def sample(self, x, w=None, p=None, posterior=False, num_samples=1, latent=False):
        """Sample from the prior or posterior.

        Args:
            x (matrix): Inputs to sample at.
            w (matrix, optional): Weights of inputs to sample at.
            p (:obj:`int`, optional): Number of outputs to sample if sampling from
                the prior.
            posterior (:obj:`bool`, optional): Sample from the prior instead of the
                posterior.
            num_samples (:obj:`int`, optional): Number of samples. Defaults to `1`.
            latent (:obj:`int`, optional): Sample the latent function instead of
                observations. Defaults to `False`.

        Returns:
            list[tensor]: Prior samples. If only a single sample is
                generated, it will be returned directly instead of in a list.
        """
        x = B.uprank(_to_torch(x))

        # Check that model is conditioned or fit if sampling from the posterior.
        if posterior and not self.is_conditioned:
            raise RuntimeError(
                "Must condition or fit model before sampling " "from the posterior."
            )
        # Check that the number of outputs is specified if sampling from the
        # prior.
        elif not posterior and p is None:
            raise ValueError("Must specify number of outputs to sample.")

        # Initialise weights.
        if w is None:
            w = B.ones(torch.float64, B.shape(x)[0], self.p if posterior else p)
        else:
            w = B.uprank(_to_torch(w))

        if posterior:
            # Construct posterior GPAR.
            gpar = _construct_gpar(self, self.vs, self.m, self.p)
            gpar = gpar | (self.x, self.y, self.w)
        else:
            # Construct prior GPAR.
            gpar = _construct_gpar(self, self.vs, B.shape(x)[1], p)

        # Construct function to undo normalisation and transformation.
        def undo_transforms(y_):
            return self._untransform_y(self._unnormalise_y(y_))

        # Perform sampling.
        samples = []
        with Counter(name="Sampling", total=num_samples) as counter:
            for _ in range(num_samples):
                counter.count()
                samples.append(
                    undo_transforms(gpar.sample(x, w, latent=latent)).detach_().numpy()
                )
        return samples[0] if num_samples == 1 else samples

    def predict(self, x, w=None, num_samples=100, latent=False, credible_bounds=False):
        """Predict at new inputs.

        Args:
            x (tensor): Inputs to predict at.
            w (tensor, optional): Weights of inputs to predict at.
            num_samples (:obj:`int`, optional): Number of samples. Defaults to `100`.
            latent (:obj:`bool`, optional): Predict the latent function instead of
                observations. Defaults to `True`.
            credible_bounds (:obj:`bool`, optional): Also return 95% central marginal
                credible bounds for the predictions.

        Returns:
            tensor: Predictive means. If `credible_bounds` is set to true,
                a three-tuple will be returned containing the predictive means,
                lower credible bounds, and upper credible bounds.
        """
        # Sample from posterior.
        samples = self.sample(
            x, w, num_samples=num_samples, latent=latent, posterior=True
        )

        # Compute mean.
        mean = np.mean(samples, axis=0)

        if credible_bounds:
            # Also return lower and upper credible bounds if asked for.
            lowers = np.percentile(samples, 2.5, axis=0)
            uppers = np.percentile(samples, 100 - 2.5, axis=0)
            return mean, lowers, uppers
        else:
            return mean