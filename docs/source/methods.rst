Methods
=======
This section lists published methods for estimation and calibration of uncertainty supported by Fortuna.

Posterior approximation methods
-------------------------------

- **Maximum-A-Posteriori (MAP)** `[Bassett et al., 2018] <https://link.springer.com/article/10.1007/s10107-018-1241-0>`_
    Approximate the posterior distribution with a Dirac delta centered at its estimated mode.
    It is the fastest ad crudest posterior approximation method supported by Fortuna. It can be seen as a form of
    regularized Maximum Likelihood Estimation (MLE) procedure.

- **Automatic Differentiation Variational Inference (ADVI)** `[Kucukelbir et al., 2017] <https://www.jmlr.org/papers/volume18/16-107/16-107.pdf>`_
    A variational inference approach that approximates the posterior distribution with a diagonal multivariate
    Gaussian distribution.

- **Laplace approximation** `[Daxberger et al., 2021] <https://proceedings.neurips.cc/paper/2021/hash/a7c9585703d275249f30a088cebba0ad-Abstract.html>`_
    The Laplace approximation approximates the posterior distribution with a Gaussian distribution. The mean is given
    by the MAP, i.e. an estimate of the mode of the posterior. The covariance matrix is expressed as the inverse of the
    Hessian of the negative-log-posterior. In practice, the Hessian is also approximated. Fortuna currently supports
    a diagonal Generalized Gauss-Newton Hessian approximation.

- **Deep ensemble** `[Lakshminarayanan et al., 2017] <https://papers.nips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html>`_
    An ensemble of Maximum-A-Posteriori (MAP) starting from different initialization, approximating the posterior
    distribution as a mixture of Dirac deltas.

- **SWAG** `[Maddox et al., 2019] <https://papers.nips.cc/paper/2019/hash/118921efba23fc329e6560b27861f0c2-Abstract.html>`_
    SWAG approximates the posterior with a Gaussian distribution. After a convergence regime is reached, the mean is
    taken by averaging checkpoints over the stochastic optimization trajectory. The covariance is also estimated
    empirically along the trajectory, and it is made of a diagonal component and a low-rank non-diagonal one.

- **Stochastic Gradient Hamiltonian Monte Carlo (SGHMC)** `[Chen et al., 2014] <http://proceedings.mlr.press/v32/cheni14.pdf>`_
    SGHMC approximates the posterior as a steady-state distribution of a Monte Carlo Markov chain with Hamiltonian dynamics.
    After the initial "burn-in" phase, each step of the chain generates samples from the posterior.

- **Cyclical Stochastic Gradient Langevin Dynamics (Cyclical SGLD)** `[Zhang et al., 2020] <https://openreview.net/pdf?id=rkeS1RVtPS>`_
    Cyclical SGLD adapts the cyclical cosine step size schedule, and alternates between *exploration* and *sampling* stages to better
    explore the multimodal posteriors for deep neural networks.

Parametric calibration methods
------------------------------
Fortuna supports parametric calibration by adding an output calibration model on top of the outputs of the model used for
training or posterior approximation, and training its parameters. A **temperature scaling** model
`[Guo et al., 2017] <https://proceedings.mlr.press/v70/guo17a.html>`_
is supported explicitly, for both classification and regression, where the outputs are calibrated using a single scaling
parameter.

Conformal prediction methods
----------------------------
We support conformal prediction methods for classification and regression.

For classification:

- **A simple conformal prediction sets method** `[Vovk et al., 2005] <https://link.springer.com/book/10.1007/b106715>`_
    A simple conformal prediction method deriving a score function from the probability associated to the largest class.

- **An adaptive conformal prediction sets method** `[Romano et al., 2020] <https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html>`_
    A method for conformal prediction deriving a score function that makes use of the full vector of class probabilities.

- **Adaptive conformal inference** `[Gibbs et al., 2021] <https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html>`_
    A method for conformal prediction that aims at correcting the coverage of conformal prediction methods in a
    sequential prediction framework (e.g. time series forecasting) when the distribution of the data shifts over time.

- **BatchMVP** `[Jung C. et al., 2022] <https://arxiv.org/pdf/2209.15145.pdf>`_
    A conformal prediction algorithm that satisfies coverage guarantees conditioned on group membership and
    non-conformity thresholds.

- **Multicalibrate** `[Hébert-Johnson Ú. et al., 2017] <https://arxiv.org/abs/1711.08513>`_, `[Roth A., Algorithm 15] <https://www.cis.upenn.edu/~aaroth/uncertainty-notes.pdf>`_
    Unlike standard conformal prediction methods, this algorithm returns scalar calibrated score values for each data point.
    For example, in binary classification, it can return calibrated probabilities of predictions.
    This method satisfies coverage guarantees conditioned on group membership and non-conformity thresholds.

For regression:

- **Conformalized quantile regression** `[Romano et al., 2019] <https://proceedings.neurips.cc/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf>`_
    A conformal prediction method that takes in input a coverage interval and calibrates it.

- **Conformal interval from scalar uncertainty measure** `[Angelopoulos et al., 2022] <https://proceedings.mlr.press/v162/angelopoulos22a.html>`_
    A conformal prediction method that takes in input a scalar measure of uncertainty (e.g. the standard deviation) and
    returns a conformal interval.

- **Jackknife+, jackknife-minmax and CV+** `[Barber et al., 2021] <https://projecteuclid.org/journals/annals-of-statistics/volume-49/issue-1/Predictive-inference-with-the-jackknife/10.1214/20-AOS1965.full>`_
    Methods based on leave-one-out and K-fold cross validation that, from model outputs only, provide conformal intervals
    satisfying minimal coverage properties.

- **BatchMVP** `[Jung C. et al., 2022] <https://arxiv.org/pdf/2209.15145.pdf>`_
    A conformal prediction algorithm that satisfies coverage guarantees conditioned on group membership and
    non-conformity thresholds.

- **EnbPI** `[Xu et al., 2021] <http://proceedings.mlr.press/v139/xu21h/xu21h.pdf>`_
    A conformal prediction method for time series regression based on data bootstrapping.

- **Multicalibrate** `[Hébert-Johnson Ú. et al., 2017] <https://arxiv.org/abs/1711.08513>`_, `[Roth A., Algorithm 15] <https://www.cis.upenn.edu/~aaroth/uncertainty-notes.pdf>`_
    Unlike standard conformal prediction methods, this algorithm returns scalar calibrated score values for each data point.
    This method satisfies coverage guarantees conditioned on group membership and non-conformity thresholds.

- **Adaptive conformal inference** `[Gibbs et al., 2021] <https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html>`_
    A method for conformal prediction that aims at correcting the coverage of conformal prediction methods in a
    sequential prediction framework (e.g. time series forecasting) when the distribution of the data shifts over time.

Out-of-distribution (OOD) detection
-----------------------------------
We support the following methods for OOD detection in classification:

- **Mahalanobis distance classifier** `[Lee et al., 2018] <https://proceedings.neurips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf>`_
    A classifier based on the Mahalanobis distance. It estimates an OOD score for each input.

- **Deep Deterministic Uncertainty (DDU)** `[Mukhoti et al., 2022] <https://arxiv.org/abs/2102.11582>`_
    Similar to the Mahalanobis distance classifier, it fits a Gaussian for each label and estimates an OOD score for each input.