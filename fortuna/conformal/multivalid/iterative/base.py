import abc
import logging
from typing import (
    Dict,
    Optional,
    Tuple,
    Union,
)

from jax import (
    random,
    vmap,
)
import jax.numpy as jnp

from fortuna.conformal.multivalid.base import MultivalidMethod
from fortuna.typing import Array


class IterativeMultivalidMethod(MultivalidMethod):
    def __init__(self, seed: int = 0):
        """
        A base iterative multivalid method.

        Parameters
        ----------
        seed: int
            Random seed.
        """
        super().__init__(seed=seed)
        self._patches = []
        self._eta = None

    def calibrate(
        self,
        scores: Array,
        groups: Optional[Array] = None,
        values: Optional[Array] = None,
        test_groups: Optional[Array] = None,
        test_values: Optional[Array] = None,
        atol: float = 1e-4,
        rtol: float = 1e-6,
        min_prob_b: float = 0.1,
        n_buckets: int = 100,
        n_rounds: int = 1000,
        eta: float = 0.1,
        split: float = 0.8,
        **kwargs,
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
        atol: float
            Absolute tolerance on the mean squared error.
        rtol: float
            Relative tolerance on the mean squared error.
        min_prob_b: Union[float, str]
            Minimum probability of the conditioning set :math:`B_t` for the patch to be applied.
            If "auto", it will be chosen based on the number of buckets and dimension of the scores.
        n_buckets: int
            The number of buckets used in the algorithm.
        n_rounds: int
            The maximum number of rounds to run the method for.
        eta: float
            Step size. By default, this is set to 1.
        split: float
            Split the calibration data into calibration and validation, according to the given proportion.
            The validation data will be used for early stopping.

        Returns
        -------
        Union[Dict, Tuple[Array, Dict]]
            A status including the number of rounds taken to reach convergence and the calibration errors computed
            during the training procedure. if `test_values` and `test_groups` are provided, the list of patches will
            be applied to `test_values`, and the calibrated test values will be returned together with the status.
        """
        if n_rounds < 1:
            raise ValueError("`n_rounds` must be at least 1.")

        if test_values is not None and values is None:
            raise ValueError(
                "If `test_values is provided, `values` must also be provided."
            )
        if values is not None and test_groups is not None and test_values is None:
            raise ValueError(
                "If `values` and `test_groups` are provided, `test_values` must also be provided."
            )
        if test_groups is not None and groups is None:
            raise ValueError(
                "If `test_groups` is provided, `groups` must also be provided."
            )
        if groups is not None and test_values is not None and test_groups is None:
            raise ValueError(
                "If `groups` and `test_values` are provided, `test_groups` must also be provided."
            )
        if eta <= 0 or eta > 1:
            raise ValueError(
                "`eta` must be a float greater than 0 and less than or equal to 1."
            )
        if split <= 0 or split >= 1:
            raise ValueError("`split` must be a float greater than 0 and less than 1.")
        if min_prob_b != "auto" and (min_prob_b < 0 or min_prob_b > 1):
            raise ValueError(
                "`min_prob_b` must be greater than or equal to 0 and less than or equal to 1."
            )

        self._check_scores(scores)
        scores = self._process_scores(scores)
        n_dims = scores.shape[1]

        min_prob_b = self._maybe_init_min_prob_b(
            min_prob_b=min_prob_b, n_buckets=n_buckets, n_dims=n_dims
        )

        groups = self._init_groups(groups, scores.shape[0])
        self._maybe_check_groups(groups, test_groups)
        n_groups = groups.shape[1]

        values = self._maybe_init_values(values, groups.shape[0])
        self._maybe_check_values(values, test_values)

        self.n_buckets = n_buckets
        buckets = self._get_buckets(n_buckets)
        values = vmap(lambda v: self._round_to_buckets(v, buckets))(values)

        self._eta = eta

        size = len(scores)
        calib_size = int(jnp.ceil(split * size))

        perm = random.choice(
            random.PRNGKey(self._seed), size, shape=(size,), replace=False
        )
        scores, val_scores = scores[perm[:calib_size]], scores[perm[calib_size:]]
        values, val_values = values[perm[:calib_size]], values[perm[calib_size:]]
        groups, val_groups = groups[perm[:calib_size]], groups[perm[calib_size:]]

        val_mses = [float(self._mean_squared_error(val_values, val_scores))]
        old_mse = float(self._mean_squared_error(values, scores))
        self._patches = []
        converged = False

        for t in range(n_rounds):
            calib_error_gvc, b = vmap(
                lambda g: vmap(
                    lambda v: vmap(
                        lambda c: self._calibration_error(
                            v=v,
                            g=g,
                            c=c,
                            scores=scores[:, c],
                            groups=groups,
                            values=values,
                            n_buckets=n_buckets,
                            **kwargs,
                        )
                    )(jnp.arange(n_dims))
                )(buckets)
            )(jnp.arange(n_groups))

            calib_error_gvc *= jnp.mean(b, axis=-1) >= min_prob_b

            gt, vt, ct, bt = self._get_gt_and_vt_and_ct_and_bt(
                calib_error_gvc=calib_error_gvc,
                buckets=buckets,
                n_groups=n_groups,
                n_dims=n_dims,
                b=b,
            )

            patch = self._get_patch(
                vt=vt,
                gt=gt,
                ct=ct,
                bt=bt,
                scores=scores[:, ct],
                groups=groups,
                values=values,
                buckets=buckets,
                **kwargs,
            )
            values = self._patch(values=values, patch=patch, bt=bt, ct=ct, eta=eta)

            val_bt = self._get_b(
                groups=val_groups,
                values=val_values,
                v=vt,
                g=gt,
                c=ct,
                n_buckets=self.n_buckets,
            )
            val_values = self._patch(
                values=val_values, patch=patch, bt=val_bt, ct=ct, eta=self.eta
            )

            mse = float(self._mean_squared_error(values, scores))
            val_mses.append(float(self._mean_squared_error(val_values, val_scores)))
            if mse > old_mse:
                logging.warning(
                    "The algorithm cannot achieve the desired tolerance. "
                    "Please try increasing `n_buckets`, or decreasing `min_prob_b`."
                )
                break
            if val_mses[-1] <= atol:
                logging.info(f"Absolute tolerance satisfied after {t} rounds.")
                converged = True
                break
            if val_mses[-1] > val_mses[-2]:
                logging.warning(f"Early stopping triggered after {t} rounds.")
                break
            if jnp.any(val_bt) and (val_mses[-2] - val_mses[-1]) / val_mses[-2] <= rtol:
                logging.info(f"Relative tolerance satisfied after {t} rounds.")
                converged = True
                break
            old_mse = jnp.copy(mse)

            self._patches.append((gt, vt, ct, patch))

        status = dict(
            n_rounds=len(self.patches),
            mean_squared_errors=val_mses,
            converged=converged,
        )

        if t == n_rounds - 1 and not converged:
            logging.warning(
                "Maximum number of rounds reached without convergence. "
                "Consider adjusting the hyperparameters. "
                "In particular: "
                "increasing `n_rounds`, increasing `n_buckets`, decreasing `min_prob_b` and increasing `eta`."
            )

        if test_groups is not None or test_values is not None:
            test_values = self.apply_patches(test_groups, test_values)
            return test_values, status
        return status

    def apply_patches(
        self,
        groups: Optional[Array] = None,
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
        if groups is None and values is None:
            raise ValueError(
                "At least one between `groups` and `values` must be provided."
            )
        if not len(self._patches):
            logging.warning("No patches available.")
            return values

        values = self._maybe_init_values(
            values, groups.shape[0] if groups is not None else None
        )
        self._maybe_check_values(values)

        groups = self._init_groups(groups, values.shape[0])
        self._maybe_check_groups(groups)

        buckets = self._get_buckets(n_buckets=self.n_buckets)
        values = vmap(lambda v: self._round_to_buckets(v, buckets))(values)

        for gt, vt, ct, patch in self._patches:
            bt = self._get_b(
                groups=groups, values=values, v=vt, g=gt, c=ct, n_buckets=self.n_buckets
            )
            values = self._patch(values=values, patch=patch, bt=bt, ct=ct, eta=self.eta)
        return values

    def calibration_error(
        self,
        scores: Array,
        groups: Optional[Array] = None,
        values: Optional[Array] = None,
        n_buckets: int = 10000,
        **kwargs,
    ) -> Array:
        """
        The reweighted average squared calibration error :math:`\mu(g) K_2(f, g, \mathcal{D})`.

        Parameters
        ----------
        scores: Array
            A score for each data point.
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
        self._check_scores(scores)
        scores = self._process_scores(scores)
        n_dims = scores.shape[1]

        self._maybe_check_groups(groups)
        self._maybe_check_values(values)

        groups = self._init_groups(groups, scores.shape[0])
        values = self._maybe_init_values(values, scores.shape[0])

        buckets = self._get_buckets(n_buckets)
        values = vmap(lambda v: self._round_to_buckets(v, buckets))(values)

        error, b = vmap(
            lambda g: vmap(
                lambda v: vmap(
                    lambda c: self._calibration_error(
                        v=v,
                        g=g,
                        c=c,
                        scores=scores[:, c],
                        groups=groups,
                        values=values,
                        n_buckets=n_buckets,
                        **kwargs,
                    )
                )(jnp.arange(n_dims))
            )(buckets)
        )(jnp.arange(groups.shape[1]))
        return error.sum(1)

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, eta):
        self._eta = eta

    def _maybe_init_values(self, values: Optional[Array], size: Optional[int] = None):
        if values is None:
            if size is None:
                raise ValueError(
                    "If `values` is not provided, `size` must be provided."
                )
            return jnp.zeros(size)
        return jnp.copy(values)

    @abc.abstractmethod
    def _calibration_error(
        self,
        v: Array,
        g: Array,
        c: Optional[Array],
        scores: Array,
        groups: Array,
        values: Array,
        n_buckets: int,
        **kwargs,
    ):
        pass

    @staticmethod
    def _get_gt_and_vt_and_ct_and_bt(
        calib_error_gvc: Array,
        buckets: Array,
        n_groups: int,
        n_dims: int,
        b: Array,
    ) -> Tuple[Array, Array, Array, Array]:
        gt, idx_vt, ct = jnp.unravel_index(
            jnp.argmax(calib_error_gvc), (n_groups, len(buckets), n_dims)
        )
        vt = buckets[idx_vt]
        return gt, vt, ct, b[gt, idx_vt, ct]

    @staticmethod
    def _get_b(
        groups: Array,
        values: Array,
        v: Array,
        g: Array,
        c: Optional[Array],
        n_buckets: int,
    ) -> Array:
        return (jnp.abs(values - v) < 0.5 / n_buckets) * groups[:, g]

    @abc.abstractmethod
    def _get_patch(
        self,
        vt: Array,
        gt: Array,
        ct: Array,
        scores: Array,
        groups: Array,
        values: Array,
        buckets: Array,
        **kwargs,
    ) -> Array:
        pass

    @staticmethod
    def _patch(values: Array, patch: Array, bt: Array, ct: Array, eta: float) -> Array:
        if values.ndim == 1:
            return values.at[bt].set(
                jnp.minimum(
                    (1 - eta) * values[bt] + eta * patch,
                    jnp.ones_like(values[bt]),
                )
            )
        return values.at[bt, ct].set(
            jnp.minimum(
                (1 - eta) * values[bt, ct] + eta * patch,
                jnp.ones_like(values[bt, ct]),
            )
        )

    @staticmethod
    def _maybe_check_groups(groups: Array, test_groups: Optional[Array] = None):
        if groups is not None and groups.ndim != 2:
            raise ValueError("`groups` must be a 2-dimensional array.")
        if test_groups is not None and test_groups.ndim != 2:
            raise ValueError("`groups` must be a 2-dimensional array.")
        if groups is not None and groups.dtype != bool:
            raise ValueError("All elements in `groups` must be a bool.")
        if test_groups is not None and test_groups.dtype != bool:
            raise ValueError("All elements in `test_groups` must be a bool.")

    @staticmethod
    def _init_groups(groups: Optional[Array], size: int) -> Array:
        if groups is None:
            return jnp.ones((size, 1), dtype=bool)
        return jnp.copy(groups)
