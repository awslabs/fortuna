from typing import (
    Callable,
    List,
    Optional,
    Tuple,
    Union,
)

from jax import vmap
import jax.numpy as jnp
import numpy as np

from fortuna.conformal.multivalid.batch_mvmp import BatchMVMPMethod
from fortuna.data.loader import (
    DataLoader,
    InputsLoader,
)
from fortuna.typing import Array


class BatchMVMPBinaryClassifier(BatchMVMPMethod):
    # def __init__(
    #     self,
    #     score_fn: Callable[[Array, Array], Array],
    #     group_fns: List[Callable[[Array], Array]],
    # ):
    #     """
    #     This class implements a classification version of BatchMVP
    #     `[Jung et al., 2022] <https://arxiv.org/abs/2209.15145>`_,
    #     a multivalid conformal prediction method that satisfies coverage guarantees conditioned on group membership
    #     and non-conformity threshold.
    #
    #     Parameters
    #     ----------
    #     score_fn: Callable[[Array, Array], Array]
    #         A score function mapping a batch of inputs and targets to scalar scores, one for each data point. The
    #         score function represents the degree of non-conformity between inputs and targets. In regression, an
    #         example of score function is :math:`s(x,y)=|y - h(x)|`, where `h` is an arbitrary regression model.
    #     group_fns: List[Callable[[Array], Array]]
    #         A list of group functions, each mapping input data points into boolean arrays which determine whether
    #         an input belongs to a certain group or not. As an example, suppose that we are interested in obtaining
    #         marginal coverages guarantee on both negative and positive scalar inputs.
    #         Then we could define groups functions
    #         :math:`g_1(x) = x < 0` and :math:`g_1(x) = x > 0`.
    #         Note that groups can be overlapping, and do not need to cover the full space of inputs.
    #     """
    #     super().__init__(score_fn=score_fn, group_fns=group_fns)
    #
    # def calibrate(
    #     self,
    #     val_data_loader: DataLoader,
    #     test_inputs_loader: InputsLoader,
    #     tol: float = 1e-4,
    #     n_rounds: int = 1000,
    #     return_max_calib_error: bool = False,
    # ) -> Union[Array, Tuple[Array, List[Array]]]:
    #     return super().calibrate(
    #         val_data_loader=val_data_loader,
    #         test_inputs_loader=test_inputs_loader,
    #         tol=tol,
    #         n_rounds=n_rounds,
    #         return_max_calib_error=return_max_calib_error
    #     )
    pass
