import flax.linen as nn
import jax.numpy as jnp

from fortuna.model.utils.spectral_norm import WithSpectralNorm


class MyModel(nn.Module):
    output_dim: int
    dense: nn.Module = nn.Dense

    @nn.compact
    def __call__(self, x, train: bool = False, **kwargs) -> jnp.ndarray:
        if hasattr(self, 'spectral_norm'):
            dense = self.spectral_norm(self.dense, train=train)
        else:
            dense = self.dense
        x = x.reshape(x.shape[0], -1)
        x = dense(2, name="l1")(x)
        x = nn.Dropout(rate=0.9)(x, deterministic=not train)
        x = dense(self.output_dim, name="l2")(x)
        return x


class MyModelWithSpectralNorm(WithSpectralNorm, MyModel):
    pass
