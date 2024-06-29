import jax
from jax import random
from jax.tree_util import (
    tree_map,
    tree_structure,
    tree_unflatten,
)
from optax._src.base import PyTree


def generate_rng_like_tree(rng, target: PyTree):
    treedef = tree_structure(target)
    keys = random.split(rng, treedef.num_leaves)
    return tree_unflatten(treedef, keys)


def generate_random_normal_like_tree(rng, target: PyTree):
    keys = generate_rng_like_tree(rng, target)
    return tree_map(
        lambda l, k: random.normal(k, l.shape, l.dtype),
        target,
        keys,
    )


class RandomNumberGenerator:
    def __init__(self, seed: int):
        """
        A random number generator object.

        Parameters
        ----------
        seed : int
            A random seed.
        """
        self._rng = random.PRNGKey(seed)

    def get(self) -> jax.Array:
        """
        Get the internal random number generator key. Whenever this function is called, the random number generator
        key is updated.

        Returns
        -------
        jax.Array
            A random number generator key.
        """
        self._rng = random.split(self._rng)[0]
        return self._rng


class WithRNG:
    @property
    def rng(self) -> RandomNumberGenerator:
        """
        Invoke the random number generator object.

        Returns
        -------
        The random number generator object.
        """
        return self._rng

    @rng.setter
    def rng(self, rng: RandomNumberGenerator):
        """
        Set a random number generator object.

        Parameters
        ----------
        rng : RandomNumberGenerator
            A random number generator object.
        """
        self._rng = rng
