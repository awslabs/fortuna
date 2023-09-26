from typing import (
    Optional,
    Union,
)

from fortuna.conformal.multivalid.mixins.classification.binary_multicalibrator import (
    BinaryClassificationMulticalibratorMixin,
)
from fortuna.conformal.multivalid.one_shot.multicalibrator import OneShotMulticalibrator
from fortuna.typing import Array


class OneShotBinaryClassificationMulticalibrator(
    BinaryClassificationMulticalibratorMixin, OneShotMulticalibrator
):
    def __init__(self, seed: int = 0):
        super().__init__(seed=seed)

    def calibrate(
        self,
        targets: Array,
        probs: Optional[Array] = None,
        test_probs: Optional[Array] = None,
        n_buckets: int = 100,
        min_prob_b: Union[float, str] = "auto",
    ):
        return super().calibrate(
            scores=targets,
            values=probs,
            test_values=test_probs,
            n_buckets=n_buckets,
            min_prob_b=min_prob_b,
        )

    def apply_patches(
        self,
        probs: Array,
    ) -> Array:
        return super().apply_patches(values=probs)
