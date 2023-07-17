# Approximate Conformal Bayes for Classification in Fortuna


## Introduction

The documentation outlines the experimental implementation of approximate [Conformal Bayes](https://proceedings.neurips.cc/paper_files/paper/2021/file/97785e0500ad16c18574c64189ccf4b4-Paper.pdf) for classification in Fortuna. Bayesian posterior predictive credible sets are not necessarily well-calibrated. Given samples from the posterior distribution, Conformal Bayes returns calibrated conformal sets based on the Bayesian model.


## Conformal Bayes
Given samples from a posterior distribution, Conformal Bayes uses importance sampling to carry out *full* conformal prediction with the posterior predictive probability as the conformity score. More details on full conformal prediction can for example be found [here](https://arxiv.org/pdf/2107.07511.pdf).

We write $Z_i = \{Y_i,X_i\}$ as our data points, where $Z_{1:n}$ is our training data and $Z_{n+1} = \{y, X_{n+1}\}$ is our test point, where $y \in \{1,\ldots,K\}$ and
$K$ is the number of classes. Our conformity measure is the posterior predictive density
$$\sigma_i(y) =  p(Y_i \mid X_i,Z_{1:n+1}) $$
for $i \in \{ 1\,\ldots,n+1\}$, where
$$p(Y_i \mid X_i,Z_{1:n+1}) = \int f_\theta(Y_i \mid X_i) \, \pi(\theta \mid Z_{1:n+1}) \, d\theta.$$

Suppose we have access to posterior samples $\theta^{(1:T)} \sim \pi(\theta \mid Z_{1:n})$. From these samples, we can also pre-computed the observed data likelihoods $f_{\theta^{(t)}}(Y_i \mid X_i)$ for $i = 1,\ldots,n$ and $t = 1,\ldots,T$. For each $y \in \{1,\ldots,K\}$, we can compute the unnormalized importance weights
$$\widetilde{w}^{(t)} = f_{\theta^{(t)}}(y \mid X_{n+1})$$
for $t\in \{1,\ldots,T\}$. We then self-normalize the weights
 $$w^{(t)} = \frac{\widetilde{w}^{(t)}}{\sum_{t'=1}^T \widetilde{w}^{(t')}}\cdot$$

 The conformity scores can then be computed as
$$
\sigma_i(y) = \sum_{t=1}^T w^{(t)} f_{\theta^{(t)}}(Y_i \mid X_i), \quad \sigma_{n+1}(y) = \sum_{t=1}^T w^{(t)} f_{\theta^{(t)}}(y \mid X_{n+1}).
$$
For error level $\alpha$, we then return $y \in \{1,\ldots,K\}$ in the conformal set if
$$
\frac{1}{n+1}\sum_{i=1}^{n+1} 1(\sigma_i(y) \leq \sigma_{n+1}(y)) > \alpha.
$$

## Approximate Conformal Bayes

In Fortuna, we instead have access to the *approximate* posterior: $\theta^{(1:T)} \sim \tilde{\pi}(\theta \mid Z_{1:n})$. One can still apply the above procedure, in the hope that the approximation is sufficiently accurate so the ranks of the residuals are unchanged.

In practice, we find that `fortuna.prob_model.LaplacePosteriorApproximator` does not work well with Conformal Bayes except for very small models. On the other hand, `fortuna.prob_model.ADVIPosteriorApproximator` works well for large range of models. We thus recommend the user stick to ADVI approximations (or MCMC) when using Conformal Bayes. Note that motivation is only heuristic for now - theoretical justification will be future work.

## Run Conformal Bayes

Conformal Bayes requires the training data loader used to train the model, and a test inputs loader at which to compute the conformal sets. We also provide the effective sample (ESS) of the importance weights to check their stability.
```
conformal_set, ESS = prob_model.predictive.conformal_set(
    train_data_loader,
    test_inputs_loader,
    error=0.2,
    n_posterior_samples=100,
    return_ess=True,
)
```

## Examples

Two example are contained in this directory:
- `example_2moons_CB.py`: 2 Moons with $n_{\text{train}}= 1000$
- `example_MNIST_CB.py`: MNIST with $n_{\text{train}}= 6000$

We compare Conformal Bayes (CB), regular Bayes and  `fortuna.conformal.AdaptivePredictionConformalClassifier` (APS).

Note that in the current setup, APS is at an advantage as we generate extra validation data to compute the conformal sets. In reality, we would need to split the training data into training/validation subsets.

We see that overall Conformal Bayes attains closer to nominal coverage and tighter intervals as a result compared to Bayes and APS.

### **Results for 2 Moons**

|  | CB  | Bayes |  APS |
|---|---|---|---|
|$1-\alpha$ = 0.9 ||||
| Mean Coverage  | 0.94 |1.0| 1.0|
| Mean Length  | 0.99 | 1.30 |1.92|
|$1-\alpha$ = 0.8 | | |
| Mean Coverage  | 0.84 |0.98 |1.0|
| Mean Length  |  0.86 | 1.11 |1.86|
|$1-\alpha$ = 0.7 | | |
| Mean Coverage  |  0.74 |0.98 |1.0|
| Mean Length  | 0.74 | 1.05| 1.81


### **Results for MNIST**

|  | CB  | Bayes |  APS |
|---|---|---|---|
|$1-\alpha$ = 0.9 ||||
| Mean Coverage  | 0.90 |0.99| 0.99|
| Mean Length  | 0.90 | 1.07 |1.24|
|$1-\alpha$ = 0.8 | | |
| Mean Coverage  | 0.79 |0.98|0.98|
| Mean Length  |  0.79 | 1.04 |1.25|
|$1-\alpha$ = 0.7 | | |
| Mean Coverage  |  0.70 | 0.98 |0.98|
| Mean Length  | 0.70 |  1.03 | 1.26|


## References

Fong, E., & Holmes, C. C. (2021). Conformal Bayesian computation. Advances in Neural Information Processing Systems, 34, 18268-18279.

Fortuna can be found [here](https://aws-fortuna.readthedocs.io/en/latest/).
