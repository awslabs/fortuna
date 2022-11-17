import flax.linen as nn
import jax.numpy as jnp


class MyModel(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x, train: bool = False, **kwargs) -> jnp.ndarray:
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(2, name="l1")(x)
        x = nn.Dropout(rate=0.9)(x, deterministic=not train)
        x = nn.Dense(self.output_dim, name="l2")(x)
        return x
