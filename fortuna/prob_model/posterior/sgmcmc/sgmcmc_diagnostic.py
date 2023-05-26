from typing import (
    List,
    Optional,
)

from jax import (
    jit,
    lax,
    vmap,
)
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from optax._src.base import PyTree


@jit
def kernel_stein_discrepancy_imq(
    samples: List[PyTree],
    grads: List[PyTree],
    c: float = 1.0,
    beta: float = -0.5,
) -> float:
    """Kernel Stein Discrepancy with the Inverse Multiquadric (IMQ) kernel.

    See `Gorham J. and Mackey L., 2017 <https://proceedings.mlr.press/v70/gorham17a/gorham17a.pdf>`_ for more details.

    Parameters
    ----------
        samples: List[PyTree]
            The list of `PyTree`, each representing an MCMC sample.
        grads: List[PyTree]
            The list of the corresponding density gradients.
        c: float
            :math:`c > 0` kernel bias hyperparameter.
        beta: float
            :math:`beta < 0` kernel exponent hyperparameter.

    Returns
    -------
        ksd_img: float
            The kernel Stein discrepancy value.
    """
    if not c > 0:
        raise ValueError("`c` should be > 0.")
    if not beta < 0:
        raise ValueError("`beta` should be < 0.")

    samples = ravel_pytree(samples)[0].reshape(len(samples), -1)
    grads = ravel_pytree(grads)[0].reshape(len(grads), -1)

    def _k_0(param1, param2, grad1, grad2, c, beta):
        dim = param1.shape[0]
        diff = param1 - param2
        base = c**2 + jnp.dot(diff, diff)
        kern = jnp.dot(grad1, grad2) * base**beta
        kern += -2 * beta * jnp.dot(grad1, diff) * base ** (beta - 1)
        kern += 2 * beta * jnp.dot(grad2, diff) * base ** (beta - 1)
        kern += -2 * dim * beta * (base ** (beta - 1))
        kern += -4 * beta * (beta - 1) * base ** (beta - 2) * jnp.sum(jnp.square(diff))
        return kern

    _batched_k_0 = vmap(_k_0, in_axes=(None, 0, None, 0, None, None))

    def _ksd(accum, x):
        sample1, grad1 = x
        accum += jnp.sum(_batched_k_0(sample1, samples, grad1, grads, c, beta))
        return accum, None

    ksd_sum, _ = lax.scan(_ksd, 0.0, (samples, grads))
    return jnp.sqrt(ksd_sum) / samples.shape[0]


def effective_sample_size(
    samples: List[PyTree], filter_threshold: Optional[float] = 0.0
) -> PyTree:
    """Estimate the effective sample size of a sequence.

    For a sequence of length :math:`N`, the effective sample size is defined as

    :math:`ESS(N) =  N / [ 1 + 2 * ( (N - 1) / N * R_1 + ... + 1 / N * R_{N-1} ) ]`

    where :math:`R_k` is the auto-correlation sequence,
    :math:`R_k := Cov{X_1, X_{1+k}} / Var{X_1}`

    Parameters
    ----------
        samples: List[PyTree]
            The list of `PyTree`, each representing an MCMC sample.
        filter_threshold: Optional[float]
            The cut-off value to truncate the sequence at the first index where
            the estimated auto-correlation is less than the threshold.

    Returns
    -------
        ESS: PyTree
            Parameter-wise estimates of the effective sample size.

    """
    unravel_fn = ravel_pytree(samples[0])[1]
    samples = ravel_pytree(samples)[0].reshape(len(samples), -1)

    def _autocorr(x, axis=-1, center=True):
        """Compute auto-correlation along one axis."""

        dtype = x.dtype
        shift = (-1 - axis) if axis < 0 else (len(x.shape) - 1 - axis)
        x = jnp.transpose(x, jnp.roll(jnp.arange(len(x.shape)), shift))
        if center:
            x -= x.mean(axis=-1, keepdims=True)

        # Zero pad to the next power of 2 greater than 2 * x_len
        x_len = x.shape[-1]
        pad_len = int(2.0 ** jnp.ceil(jnp.log2(x_len * 2)) - x_len)
        x = jnp.pad(x, (0, pad_len))[:-pad_len]

        # Autocorrelation is IFFT of power-spectral density
        fft = jnp.fft.fft(x.astype(jnp.complex64))
        prod = jnp.fft.ifft(fft * jnp.conj(fft))
        prod = jnp.real(prod[..., :x_len]).astype(dtype)

        # Divide to obtain an unbiased estimate of the expectation
        denominator = x_len - jnp.arange(0.0, x_len)
        res = prod / denominator
        return jnp.transpose(res, jnp.roll(jnp.arange(len(res.shape)), -shift))

    auto_cov = _autocorr(samples, axis=0)
    auto_corr = auto_cov / auto_cov[:1]

    n = len(samples)
    nk_factor = (n - jnp.arange(0.0, n)) / n
    weighted_auto_corr = nk_factor[..., None] * auto_corr

    if filter_threshold is not None:
        mask = (auto_corr < filter_threshold).astype(auto_corr.dtype)
        mask = jnp.cumsum(mask, axis=0)
        mask = jnp.maximum(1.0 - mask, 0.0)
        weighted_auto_corr *= mask

    ess = n / (-1 + 2 * weighted_auto_corr.sum(axis=0))
    return unravel_fn(ess)
