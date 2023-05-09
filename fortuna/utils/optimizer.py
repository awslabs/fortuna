import optax
from flax import traverse_util
from flax.core import FrozenDict
from typing import Dict, Optional, Callable

import jax.numpy as jnp
from optax._src import base

from fortuna.typing import Params


def stepped_schedule(
    init_value: float, boundaries_and_scales: Optional[Dict[int, float]] = None
) -> base.Schedule:
    """
    Returns a stepped learning rate schedule with optional warmup.

    A stepped learning rate schedule decreases the learning rate
    by specified amounts at specified epochs.
    A common ImageNet schedule decays the learning rate by a factor of 0.1 at epochs 30, 60 and 80.
    This would be specified setting `boundaries_and_scales` to:

      {
        30: 0.1,
        60: 0.01,
        80: 0.001
      }

    Parameters
    ----------
    init_value: float
        An initial value `init_v`.
    boundaries_and_scales: Optional[Dict[int, float]]
        The schedule as a dict like `{epoch_x: lr_factor_x, epoch_y: lr_factor_y}`;
    the step occurs at epoch `epoch` and sets the learning rate to `init_value * lr_factor`

    Returns
    -------
    base.Schedule
        A function that maps step counts to values.
    """
    if boundaries_and_scales is not None:
        all_positive = all(scale >= 0.0 for scale in boundaries_and_scales.values())
        if not all_positive:
            raise ValueError(
                "`piecewise_constant_schedule` expects non-negative scale factors"
            )

    def schedule(count):
        v = init_value
        if boundaries_and_scales is not None:
            for threshold, scale in sorted(boundaries_and_scales.items()):
                indicator = jnp.maximum(0.0, jnp.sign(threshold - count))
                v = v * indicator + (1 - indicator) * scale * init_value
        return v

    return schedule


def linear_scheduler_with_warmup(
        learning_rate: float,
        num_inputs_train: int,
        train_total_batch_size: int,
        num_train_epochs: int,
        num_warmup_steps: int,
) -> Callable[[int], jnp.array]:
    """
    Create a linear scheduler with a warmup

    Parameters
    ----------
    learning_rate: float
        Learning rate value
    num_inputs_train: int
        Number of input data points in the training set.
    train_total_batch_size: int
        Total training batch size.
    num_train_epochs: int
        Number of training epochs.
    num_warmup_steps: int
        Number of warmup steps. At the end of the warm-up phase the learning rate value will be `learning_rate`.

    Returns
    -------
    Callable[[int], jnp.array]
        The scheduler function
    """
    steps_per_epoch = num_inputs_train // train_total_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def decay_mask_without_layer_norm_fn(params: Params) -> Params:
    flat_params = traverse_util.flatten_dict(params)
    # find out all LayerNorm parameters
    layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
    layer_norm_named_params = {
        layer[-2:]
        for layer_norm_name in layer_norm_candidates
        for layer in flat_params.keys()
        if layer_norm_name in "".join(layer).lower()
    }
    flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
    return FrozenDict(traverse_util.unflatten_dict(flat_mask))
