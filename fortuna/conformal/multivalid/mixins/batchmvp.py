from typing import Optional

import jax.numpy as jnp

from fortuna.typing import Array


class BatchMVPMixin:
    def pinball_loss(self, values: Array, scores: Array, coverage: float) -> Array:
        """
        The pinball loss between the model evaluations and the scores.

        Parameters
        ----------
        values: Array
            The model evaluations.
        scores: Array
            The scores.
        coverage: float
            The target coverage.

        Returns
        -------
        Array
            The pinball loss evaluation.
        """
        return self._loss_fn(values, scores, coverage=coverage)

    def _loss_fn(
        self, values: Array, scores: Array, coverage: Optional[float] = None
    ) -> Array:
        if scores.ndim == 2 and values.ndim == 1:
            scores = scores[:, 0]
        if coverage is None:
            coverage = self._coverage
        diff = scores - values
        return jnp.mean(
            diff * coverage * (scores > values)
            - diff * (1 - coverage) * (scores <= values)
        )
