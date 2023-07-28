import abc
import logging
from typing import (
    Callable,
    List,
    Optional,
    Tuple,
    Union,
    Dict
)

from jax import vmap
import jax.numpy as jnp

from fortuna.data.loader import (
    DataLoader,
    InputsLoader,
)
from fortuna.typing import Array


class Normalizer:
    def __init__(self, x_min: Union[float, Array], x_max: Union[float, Array]):
        self.x_min = x_min
        self.x_max = x_max if x_max != x_min else x_min + 1

    def normalize(self, x: Array) -> Array:
        return (x - self.x_min) / (self.x_max - self.x_min)

    def unnormalize(self, y: Array) -> Array:
        return self.x_min + (self.x_max - self.x_min) * y


class Model:
    def __init__(self, model_fn: Callable[[Array], Array]):
        self.model_fn = model_fn

    def __call__(self, x: Array):
        v = self.model_fn(x)
        if v.ndim > 1:
            raise ValueError(
                "Evaluations of the model function `model_fn` must be one-dimensional arrays, "
                f"but its shape was {v.shape}."
            )
        if jnp.any(v < 0) or jnp.any(v > 1):
            raise ValueError("The model function must take values within [0, 1].")
        return v


class MultivalidMethod:
    def __init__(self):
        self._patches = []
        self._n_buckets = None

    def calibrate(
        self,
        scores: Array,
        groups: Array,
        values: Optional[Array] = None,
        test_groups: Optional[Array] = None,
        test_values: Optional[Array] = None,
        tol: float = 1e-4,
        n_buckets: int = None,
        n_rounds: int = 1000,
        **kwargs
    ) -> Union[Dict, Tuple[Array, Dict]]:
        """
        Calibrate the model by finding a list of patches to the model that bring the calibration error below a
        certain threshold.

        Parameters
        ----------
        scores: Array
            A list of scores :math:`s(x, y)` computed on the calibration data.
            This should be a one-dimensional array of elements between 0 and 1.
        groups: Array
            A list of groups :math:`g(x)` computed on the calibration data.
            This should be a two-dimensional array of bool elements.
            The first dimension is over the data points, the second dimension is over the number of groups.
        values: Optional[Array]
            The initial model evalutions :math:`f(x)` on the calibration data. If not provided, these are set to 0.
        test_groups: Optional[Array]
            A list of groups :math:`g(x)` computed on the test data.
            This should be a two-dimensional array of bool elements.
            The first dimension is over the data points, the second dimension is over the number of groups.
        test_values: Optional[Array]
            The initial model evaluations :math:`f(x)` on the test data. If not provided, these are set to 0.
        tol: float
            A tolerance on the reweighted average squared calibration error, i.e. :math:`\mu(g) K_2(f, g, \mathcal{D})`.
        n_buckets: int
            The number of buckets used in the algorithm. The smaller the number of buckets, the simpler the model,
            the better its generalization abilities. If not provided, We start from 2 buckets, and progressively double
            the number of buckets until we find a value for which the calibration error falls below the given
            tolerance. Such number of buckets is guaranteed to exist.
        n_rounds: int
            The maximum number of rounds to run the method for.
        Returns
        -------
        Union[Dict, Tuple[Array, Dict]]
            A status including the number of rounds taken to reach convergence and the calibration errors computed
            during the training procedure. if `test_values` and `test_groups` are provided, the list of patches will
            be applied to `test_values`, and the calibrated test values will be returned together with the status.
        """
        if tol >= 1:
            raise ValueError("`tol` must be smaller than 1.")
        if n_rounds < 1:
           raise ValueError("`n_rounds` must be at least 1.")

        if test_groups is None and test_values is not None:
            raise ValueError("If `test_values` is provided, `test_groups` must be also provided.")
        if test_values is not None and values is None:
            raise ValueError("If `test_values is provided, `values` must also be provided.")
        if values is not None and test_groups is not None and test_values is None:
            raise ValueError("If `values` and `test_groups` are provided, `test_values` must also be provided.")

        if values is None:
            values_init = jnp.zeros(groups.shape[0])
        else:
            values_init = jnp.copy(values)

        self._check_range(dict(scores=scores, values=values, test_values=test_values))

        increase_n_buckets = False
        if n_buckets is None:
            n_buckets = 2
            increase_n_buckets = True

        n_groups = groups.shape[1]
        tol_reached = False

        while True:
            logging.info(f"Attempt reaching tolerance with {n_buckets} buckets.")
            buckets = self._get_buckets(n_buckets)
            values = vmap(lambda v: self._round_to_buckets(v, buckets))(values_init)

            max_calib_errors = []
            old_calib_errors_vg = None
            self._patches = []

            for t in range(n_rounds):
                calib_error_vg = vmap(
                    lambda g: vmap(
                        lambda v: self._calibration_error(
                            v,
                            g,
                            scores=scores,
                            groups=groups,
                            values=values,
                            n_buckets=n_buckets,
                            **kwargs,
                        )
                    )(buckets)
                )(jnp.arange(n_groups))

                max_calib_errors.append(calib_error_vg.sum(1).max())
                if max_calib_errors[-1] <= tol:
                    tol_reached = True
                    logging.info(f"Tolerance satisfied after {t} rounds with {n_buckets} buckets.")
                    break
                if old_calib_errors_vg is not None and jnp.allclose(old_calib_errors_vg, calib_error_vg):
                    break
                old_calib_errors_vg = jnp.copy(calib_error_vg)

                gt, vt = self._get_gt_and_vt(
                    calib_error_vg=calib_error_vg, buckets=buckets, n_groups=n_groups
                )
                bt = self._get_b(
                    groups=groups, values=values, v=vt, g=gt, n_buckets=len(buckets)
                )
                patch = self._get_patch(
                    vt=vt, gt=gt, scores=scores, groups=groups, values=values, buckets=buckets, **kwargs
                )
                values = self._patch(values=values, patch=patch, bt=bt)

                self._patches.append((gt, vt, patch))

            if tol_reached:
                break
            if increase_n_buckets:
                n_buckets *= 2
            else:
                break

        self.n_buckets = n_buckets
        status = dict(n_rounds=len(self.patches), max_calib_errors=max_calib_errors)

        if test_groups is not None:
            test_values = self.apply_patches(test_groups, test_values)
            return test_values, status
        return status

    def apply_patches(
        self,
        groups: Array,
        values: Optional[Array] = None,
    ) -> Array:
        """
        Apply the patches to the model evaluations.

        Parameters
        ----------
        groups: Array
            A list of groups :math:`g(x)` evaluated over some inputs.
            This should be a two-dimensional array of bool elements.
            The first dimension is over the data points, the second dimension is over the number of groups.
        values: Optional[Array]
            The initial model evaluations :math:`f(x)` evaluated over some inputs. If not provided, these are set to 0.

        Returns
        -------
        Array
            The calibrated values.
        """
        if not len(self._patches):
            logging.warning("No patches available.")
            return values
        if values is None:
            values = jnp.zeros(groups.shape[0])

        buckets = self._get_buckets(n_buckets=self.n_buckets)
        values = vmap(lambda v: self._round_to_buckets(v, buckets))(values)

        for gt, vt, patch in self._patches:
            bt = self._get_b(groups=groups, values=values, v=vt, g=gt, n_buckets=self.n_buckets)
            values = self._patch(values=values, bt=bt, patch=patch)
        return values

    def calibration_error(
        self,
        scores: Array,
        groups: Array,
        values: Array,
        n_buckets: int = 10000,
        **kwargs
    ) -> Array:
        """
        The reweighted average squared calibration error :math:`\mu(g) K_2(f, g, \mathcal{D})`.

        Parameters
        ----------
        scores
        groups: Array
            A list of groups :math:`g(x)` evaluated over some inputs.
            This should be a two-dimensional array of bool elements.
            The first dimension is over the data points, the second dimension is over the number of groups.
        values: Array
            The model evaluations, before or after calibration.
        n_buckets: int
            The number of buckets used in the algorithm.

        Returns
        -------
        Array
            The computed calibration error for each group
        """
        buckets = self._get_buckets(n_buckets)
        values = vmap(lambda v: self._round_to_buckets(v, buckets))(values)

        return vmap(
            lambda g: vmap(
                lambda v: self._calibration_error(
                    v=v, g=g, scores=scores, groups=groups, values=values, n_buckets=n_buckets, **kwargs
                )
            )(buckets)
        )(jnp.arange(groups.shape[1])).sum(1)
    
    @property
    def patches(self):
        return self._patches
    
    @property
    def n_buckets(self):
        return self._n_buckets

    @n_buckets.setter
    def n_buckets(self, n_buckets):
        self._n_buckets = n_buckets

    @abc.abstractmethod
    def _calibration_error(
        self,
        v: float,
        g: Array,
        scores: Array,
        groups: Array,
        values: Array,
        n_buckets: int,
        **kwargs
    ):
        pass

    @staticmethod
    def _init_missing_model_fn():
        return Model(lambda x: jnp.zeros(x.shape[0]))

    @staticmethod
    def _get_gt_and_vt(
        calib_error_vg: Array, buckets: Array, n_groups: int
    ) -> Tuple[Array, Array]:
        gt, idx_vt = jnp.unravel_index(
            jnp.argmax(calib_error_vg), (n_groups, len(buckets))
        )
        vt = buckets[idx_vt]
        return gt, vt

    @staticmethod
    def _get_b(
        groups: Array, values: Array, v: Array, g: Array, n_buckets: int
    ) -> Array:
        return (jnp.abs(values - v) < 0.5 / n_buckets) * groups[:, g]

    @abc.abstractmethod
    def _get_patch(
        self,
        vt: Array,
        gt: Array,
        scores: Array,
        groups: Array,
        values: Array,
        buckets: Array,
        **kwargs
    ) -> Array:
        pass

    @staticmethod
    def _patch(values: Array, patch: Array, bt: Array, _shift: bool = False) -> Array:
        return values.at[bt].set(
            jnp.minimum(
                patch if not _shift else values[bt] + patch,
                jnp.ones_like(values[bt]),
            )
        )

    @staticmethod
    def _get_buckets(n_buckets: int):
        return jnp.linspace(0, 1, n_buckets)

    @staticmethod
    def _round_to_buckets(v: Array, buckets: Array):
        return buckets[jnp.argmin(jnp.abs(v - buckets))]

    @staticmethod
    def _check_range(d):
        def _maybe_check(k, v):
            if v is not None:
                if v.min() < 0 or v.max() > 1:
                    raise ValueError(f"All elements in `{k}` must be between 0 and 1.")
        for k, v in d.items():
            _maybe_check(k, v)
