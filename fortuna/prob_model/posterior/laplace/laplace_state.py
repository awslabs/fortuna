from __future__ import annotations

from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import jax.numpy as jnp
from flax.core import FrozenDict

from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.typing import (
    AnyKey,
    Array,
    Params,
)
from fortuna.utils.nested_dicts import nested_pair
from fortuna.utils.strings import (
    convert_string_to_jnp_array,
    encode_tuple_of_lists_of_strings_to_numpy,
)


class LaplaceState(PosteriorState):
    """
    Attributes
    ----------
    prior_log_var: float
        Prior log-variance value.
    encoded_name: jnp.ndarray
        Laplace state name encoded as an array.
    """

    prior_log_var: float = 0.0
    encoded_name: jnp.ndarray = convert_string_to_jnp_array("LaplaceState")
    _encoded_which_params: Optional[Dict[str, Array]] = None

    @classmethod
    def convert_from_map_state(
        cls,
        map_state: MAPState,
        hess_lik_diag: Union[Params, Tuple[Params, ...]],
        prior_log_var: Optional[float],
        which_params: Tuple[List[AnyKey], ...],
    ) -> LaplaceState:
        """
        Convert a MAP state into a Laplace state.

        Parameters
        ----------
        map_state: MAPState
            A MAP state.
        hess_lik_diag: Union[Params, Tuple[Params, ...]]
            Diagonal of the approximated Hessian of the likelihood.
        prior_log_var: float
            Prior log-variance value. If None, initialize it to 100.
        which_params: Tuple[List[AnyKey], ...]
            Sequences of keys pointing to the parameters over which `std` is defined. If `which_params` is None,
            `std` must be defined for all parameters.

        Returns
        -------
        LaplaceState
            A Laplace state instance.
        """
        params = map_state.params.unfreeze()
        if which_params is not None:
            params = nested_pair(
                d=params,
                key_paths=which_params,
                objs=hess_lik_diag,
                labels=("mean", "hess_lik_diag"),
            )
        else:
            for k, v in params.items():
                params[k] = FrozenDict(
                    {
                        "params": dict(
                            mean=v["params"], hess_lik_diag=hess_lik_diag[k]["params"]
                        )
                    }
                )
        d = vars(
            map_state.replace(
                params=FrozenDict(params), encoded_name=LaplaceState.encoded_name
            )
        )
        d["_encoded_which_params"] = encode_tuple_of_lists_of_strings_to_numpy(
            which_params
        )
        d["prior_log_var"] = prior_log_var if prior_log_var is not None else 0.0
        return LaplaceState.init_from_dict(d)
