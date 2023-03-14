from .kernel import AbstractKernel
import jax
import jax.numpy as jnp
import flax.linen as nn


class Gaussian(nn.Module):
    variance: jax.Array = jnp.array([1.0])

    @nn.compact
    def __call__(self, x):
        variance = self.param("sigma2", nn.initializers.ones_init(), (1,))
        noisy_diagonal = jnp.diag(x) + variance
        diag_elements = jnp.diag_indices_from(x)
        return x.at[diag_elements].set(noisy_diagonal)


class GP(nn.Module):
    kernel: AbstractKernel
    likelihood: Gaussian = Gaussian()

    @nn.compact
    def __call__(self, xtrain, ytrain, xtest):
        # TODO: How do we compute the kernel here?
        Kff = self.kernel.apply(xtrain)  # pseudo
        # Stabilise the covariance matrix
        Kff += jnp.eye(xtrain.shape[0]) * 1e-6

        # Add noise
        noisyKff = self.likelihood.apply(Kff)  # pseudo

        Kxx = self.kernel.apply(xtest)  # pseudo
        Kxf = self.kernel.apply(xtest, xtrain)  # pseudo

        sigma_inverse_kxt = jax.scipy.linalg.solve(noisyKff, Kxf)

        # Compute the posterior mean
        # TODO: Zero-mean OK? Should we implement mean functions too?
        posterior_mean = jnp.matmul(
            jnp.transpose(Kxf), ytrain
        )  # We're assuming a zero-mean here.

        # Compute the posterior covariance
        posterior_covariance = Kxx - jnp.matmul(
            jnp.transpose(Kxf), sigma_inverse_kxt
        )
        # Stabilise
        posterior_covariance += jnp.eye(xtest.shape[0]) * 1e-6

        # TODO: What do we wish to return here? The latent mean and covariance?
        # TODO: The predictive mean and covariance i.e., self.likelihood.apply(posterior_covariance)?
        # TODO: A Gaussian distribution? If so, from what library?
        return posterior_mean, self.likelihood.apply(posterior_covariance)


class MarginalLogLikelihood(nn.Module):
    posterior: GP

    @nn.compact
    def __call__(self, xtrain, ytrain):
        n_data = xtrain.shape[0]

        # Compute the covariance
        Kff = self.kernel.apply(xtrain)  # pseudo
        # Stabilise the covariance matrix
        Kff += jnp.eye(xtrain.shape[0]) * 1e-6

        # Add noise
        noisyKff = self.likelihood.apply(Kff)  # pseudo

        # Assuming a zero-mean. With a meanf we'd subtract from y here
        delta = ytrain

        constant = n_data * jnp.log(2 * jnp.pi)
        function_complexity = jnp.linalg.slogdet(noisyKff)[1]
        data_fit = jnp.matmul(
            jnp.transpose(delta), jax.scipy.linalg.solve(noisyKff, delta)
        )
        mll = -0.5 * (constant + function_complexity + data_fit)
        return mll
