import unittest

from flax.core import FrozenDict
from jax import grad
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import optax

from fortuna.utils.freeze import (
    all_values_in_labels,
    freeze_optimizer,
    get_trainable_paths,
)


class TestFreeze(unittest.TestCase):
    def test_all_values_in_labels(self):
        self.assertRaises(ValueError, all_values_in_labels, [1, 2, 3], [3, 2])
        all_values_in_labels([3, 2], [1, 1, 4, 3, 2])

    def test_get_trainable_paths(self):
        params = FrozenDict(
            dict(a=dict(b=dict(c=[1, 2], d=dict(e=6)), f=dict(g=[4, 5])), h=dict(i=3))
        )
        paths = get_trainable_paths(
            params=params,
            freeze_fun=lambda p, v: "trainable"
            if ("b" in p and "d" in p) or "h" in p
            else "frozen",
        )
        assert paths == (["a", "b", "d", "e"], ["h", "i"])

        self.assertRaises(
            ValueError,
            get_trainable_paths,
            params,
            freeze_fun=lambda p, v: True
            if ("b" in p and "d" in p) or "h" in p
            else False,
        )

    def test_freeze_optimizer(self):
        params = FrozenDict(
            dict(
                a=dict(
                    b=dict(c=jnp.array([1.0, 2.0]), d=dict(e=jnp.array(6.0))),
                    f=dict(g=jnp.array([4.0, 5.0])),
                ),
                h=dict(i=jnp.array(3.0)),
            )
        )
        optimizer = freeze_optimizer(
            params=params,
            optimizer=optax.adam(1e-1),
            freeze_fun=lambda p, v: "trainable"
            if ("b" in p and "d" in p) or "h" in p
            else "frozen",
        )
        loss_fn = lambda p: jnp.mean((ravel_pytree(p)[0] - 1.5) ** 2)
        grads = grad(loss_fn)(params)
        opt_state = optimizer.init(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updated_params = optax.apply_updates(params, updates)
        assert jnp.all(params["a"]["b"]["c"] == updated_params["a"]["b"]["c"])
        assert jnp.all(params["a"]["b"]["d"] != updated_params["a"]["b"]["d"])
        assert jnp.all(params["a"]["f"]["g"] == updated_params["a"]["f"]["g"])
        assert jnp.all(params["h"] != updated_params["h"])
