import logging

from lab import B
from stheno import GP, Obs, SparseObs, WeightedUnique
from plum import Dispatcher

__all__ = ["GPAR"]

log = logging.getLogger(__name__)

_dispatch = Dispatcher()


def merge(x, updates, to_update):
    """Merge updates into a tensor.

    Args:
        x (tensor): Tensor to merge updates into.
        updates (tensor): Updates.
        to_update (tensor): A boolean array indicating which elements of `x` to
            update and replace with the corresponding element in `updates`.

    Returns:
        tensor: Updated tensor.
    """
    # Stack them, which screws up the order.
    concat = B.concat(x[~to_update], updates, axis=0)

    # Generate an index mapping to fix the ordering.
    i_original = 0
    i_update = B.sum(~to_update)
    indices = []
    for i in range(len(to_update)):
        # Careful not to update the indices in-place! That generates trouble
        # with PyTorch.
        if to_update[i]:
            indices.append(i_update)
            i_update = i_update + 1
        else:
            indices.append(i_original)
            i_original = i_original + 1

    # Perform the fix.
    return B.take(concat, indices)


def construct_model(f, e):
    """Convenience function that returns a model constructor.

    Args:
        f (:class:`stheno.GP`): Latent process.
        e (:class:`stheno.GP`): Noise process.

    Returns:
        function: Model constructor.
    """
    return lambda: (f, e)


def last(xs, select=None):
    """Zip a list with a boolean indicating whether it is the last element.

    Args:
        xs (list): List to zip.
        select (list[int], optional): Select particular indices from the result.
            Defaults to returning everything.

    Returns:
        list: `xs` zipped with a boolean on the left.
    """
    # Convert `select` to a set if it is given.
    if select is not None:
        select = set(select)

    # Initialise tracking variables.
    saved_x = None
    i = -1

    # Construct a function that determines the element with index `i` should be
    # yielded.
    def should_yield(i_):
        return i >= 0 and (select is None or i_ in select)

    # Loop through `xs` and check to yield every *previous* element.
    for x in xs:
        if should_yield(i):
            yield False, saved_x
        saved_x = x
        i += 1

    # Check to yield the last element, if there is one.
    if saved_x is not None and should_yield(i):
        yield True, saved_x


class GPAR:
    """Basic GPAR model.

    Args:
        replace (bool, optional): Condition on the predictive mean instead of the
            data. Defaults to `False`.
        impute (bool, optional): Impute missing data points with the predictive mean
            to make the data set closed downwards. Defaults to `False`.
        x_ind (tensor, optional): Locations of inducing points for a sparse
            approximation. Defaults to `None`.

    Attributes:
        replace (bool): See the argument `replace`.
        impute (bool): See the argument `impute`.
        layers (list): Layers of the model.
        sparse (bool): Use a sparse approximation?
        x_ind (tensor or `None`): Locations of inducing points if a sparse
            approximation is used.
    """

    def __init__(self, replace=False, impute=False, x_ind=None):
        self.replace = replace
        self.impute = impute
        self.layers = []

        # Parse inputs of inducing points.
        self.sparse = x_ind is not None
        self.x_ind = None if x_ind is None else x_ind

    def copy(self):
        """Create a new GPAR model with the same configuration.

        Returns:
            :class:`.gpar.GPAR`: New GPAR model with the same configuration.
        """
        gpar = GPAR(replace=self.replace, impute=self.impute, x_ind=self.x_ind)
        return gpar

    def add_layer(self, model_constructor):
        """Add a layer.

        Args:
            model_constructor (function): Constructor of the model, which should
                return a tuple containing the latent and noise process.

        Returns:
            :class:`.gpar.GPAR`: Updated GPAR model.
        """
        gpar = self.copy()
        gpar.layers = list(self.layers) + [model_constructor]
        return gpar

    def __or__(self, x_y_w):
        """Condition on data.

        Args:
            x (tensor): Inputs.
            y (tensor): Outputs.
            w (tensor): Weights.

        Returns:
            :class:`.gpar.GPAR`: Updated GPAR model.
        """
        x, y, w = x_y_w
        gpar, x_ind = self.copy(), self.x_ind

        for is_last, ((y, w, mask), model) in last(
            zip(per_output(y, w, keep=self.impute), self.layers)
        ):
            x = x[mask]  # Filter according to mask.
            f, e = model()  # Construct model.
            obs = self._obs(x, x_ind, y, w, f, e)  # Construct observations.

            # Update with posterior.
            post = f.measure | obs
            e_new = GP(e.mean, e.kernel, measure=post)
            gpar.layers.append(construct_model(post(f), e_new))

            # Update inputs.
            if not is_last:
                x, x_ind = self._update_inputs(x, x_ind, y, f, obs)

        return gpar

    def logpdf(
        self,
        x,
        y,
        w,
        only_last_layer=False,
        sample_missing=False,
        return_inputs=False,
        x_ind=None,
        outputs=None,
    ):
        """Compute the logpdf.

        Args:
            x (tensor): Inputs.
            y (tensor): Outputs.
            w (tensor): Weights.
            only_last_layer (:obj:`bool`, optional): Compute the logpdf for only the last
                layer. Defaults to `False`.
            sample_missing (:obj:`bool`, optional): Sample missing data to compute an
                unbiased estimate of the pdf, *not* logpdf. Defaults to `False`.
            return_inputs (:obj:`bool`, optional): Instead return the inputs and inputs for
                the inducing points with previous outputs concatenated. This can be used
                to perform precomputation. Defaults to `False`.
            x_ind (tensor, optional): Inputs for the inducing points. This can be
                used to resume a computation. Defaults to :attr:`.model.GPAR.x_ind`.
            outputs (:obj:`list[int]`, optional): Only compute the logpdf for a subset of
                outputs. The list specifies the indices of the outputs. Defaults to
                computing the logpdf for all outputs.

        Returns:
            scalar: Logpdf. If `return_inputs` is set to `True`, instead return a
                tuple containing the inputs and the inputs for the inducing points
                with previous outputs concatenated
        """
        logpdf = B.cast(B.dtype(x), 0)
        x_ind = self.x_ind if x_ind is None else x_ind

        y_per_output = per_output(y, w, keep=self.impute or sample_missing)
        for is_last, ((y, w, mask), model) in last(
            zip(y_per_output, self.layers), select=outputs
        ):
            x = x[mask]  # Filter according to mask.
            f, e = model()  # Construct model.
            obs = self._obs(x, x_ind, y, w, f, e)  # Construct observations.

            # Accumulate logpdf.
            if not only_last_layer or (is_last and only_last_layer):
                logpdf = logpdf + f.measure.logpdf(obs)

            if not is_last:
                missing = B.isnan(y[:, 0])
                # Sample missing data for an unbiased sample of the pdf.
                if sample_missing and B.any(missing):
                    post = f.measure | obs
                    x_missing_weighted = WeightedUnique(x[missing], w[missing])
                    y = merge(y, post(f + e)(x_missing_weighted).sample(), missing)

                # Update inputs.
                x, x_ind = self._update_inputs(x, x_ind, y, f, obs)

        # Return inputs if asked for.
        return (x, x_ind) if return_inputs else logpdf

    def sample(self, x, w, latent=False):
        """Sample.

        Args:
            x (tensor): Inputs to sample at.
            w (tensor): Weights.
            latent (:obj:`bool`, optional): Sample latent function. Defaults to `False`.

        Returns:
            tensor: Sample.
        """
        sample = B.zeros(B.dtype(x), B.shape(x)[0], 0)
        x_ind = self.x_ind

        for i, (is_last, model) in enumerate(last(self.layers)):
            f, e = model()  # Construct model.

            x_weighted = WeightedUnique(x, w[:, i])

            if latent:
                # Sample latent function: use ancestral sampling.
                f_sample = f(x_weighted).sample()
                y_sample = f_sample + e(x_weighted).sample()
                sample = B.concat(sample, f_sample, axis=1)
            else:
                # Sample observed function.
                y_sample = (f + e)(x_weighted).sample()
                sample = B.concat(sample, y_sample, axis=1)

            # Update inputs.
            if not is_last:
                x, x_ind = self._update_inputs(x, x_ind, y_sample, f, None)

        return sample

    def _obs(self, x, x_ind, y, w, f, e):
        # Filter available data points.
        available = ~B.isnan(y[:, 0])
        x = x[available]
        y = y[available]
        w = w[available]

        # Perform weighting.
        x_weighted = WeightedUnique(x, w)

        if self.sparse:
            return SparseObs(f(x_ind), e, f(x_weighted), y)
        else:
            return Obs((f + e)(x_weighted), y)

    def _update_inputs(self, x, x_ind, y, f, obs):
        available = ~B.isnan(y[:, 0])

        def estimate(x_):
            # If observations are available, estimate using the posterior mean;
            # otherwise, use the prior mean.
            if obs:
                post = f.measure | obs
                return post(f).mean(x_)
            else:
                return f.mean(x_)

        # Update inputs of inducing points.
        if self.sparse:
            x_ind = B.concat(x_ind, estimate(x_ind), axis=1)

        # Impute missing data and replace available data.
        if self.impute and self.replace:
            y = estimate(x)
        else:
            # Just impute missing data.
            if self.impute and B.any(~available):
                y = merge(y, estimate(x[~available]), ~available)

            # Just replace available data.
            if self.replace and B.any(available):
                y = merge(y, estimate(x[available]), available)

        # Finally, actually update inputs.
        x = B.concat(x, y, axis=1)

        return B.dense(x), B.dense(x_ind)


@_dispatch
def per_output(y: B.Numeric, w: B.Numeric, keep=False):
    """Return observations per output, respecting that the data must be
    closed downwards.

    The function supports caching by feeding it a dictionary where the keys the are
    values for `keep` and the values lists containing items that the function should
    yield.

    Args:
        y (tensor): Outputs.
        w (tensor): Weights.
        keep (bool, optional): Also return missing observations that would make the
            data closed downwards.

    Returns:
        generator: Generator that generates tuples containing the observations per
            layer and a mask which observations are not missing relative to the
            previous layer.
    """
    p = B.shape(y)[1]  # Number of outputs
    available = ~B.isnan(y)  # Availability of outputs.

    for i in range(p):
        # Initialise the mask to current availability.
        mask = available[:, i]

        # Take into account future observations if necessary.
        if keep and i < p - 1:  # Check whether this is the last output.
            mask = mask | B.any(available[:, i + 1 :], axis=1)

        # Give stuff back.
        yield y[mask, i : i + 1], w[mask, i], mask

        # Filter observations and availability.
        y = y[mask]
        w = w[mask]
        available = available[mask]


@_dispatch
def per_output(cache: dict, _: type(None), keep=False):
    for yi in cache[keep]:
        yield yi