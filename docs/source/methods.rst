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

- **An adaptive conformal prediction sets** `[Romano et al., 2020] <https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html>`_
    A method for conformal prediction deriving a score function that makes use of the full vector of class probabilities.

For regression:

- **Conformalized quantile regression** `[Romano et al., 2019] <https://proceedings.neurips.cc/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf>`_
    A conformal prediction method that takes in input a coverage interval and calibrates it.

- **Conformal interval from scalar uncertainty measure** `[Angelopoulos et al., 2022] <https://proceedings.mlr.press/v162/angelopoulos22a.html>`_
    A conformal prediction method that takes in input a scalar measure of uncertainty (e.g. the standard deviation) and
    returns a conformal interval.

- **Jackknife+, jackknife-minmax and CV+** `[Barber et al., 2021] <https://projecteuclid.org/journals/annals-of-statistics/volume-49/issue-1/Predictive-inference-with-the-jackknife/10.1214/20-AOS1965.full>`_
    Methods based on leave-one-out and K-fold cross validation that, from model outputs only, provide conformal intervals
    satisfying minimal coverage properties.

- **EnbPI** `[Xu et al., 2021] <http://proceedings.mlr.press/v139/xu21h/xu21h.pdf>`_
    A conformal prediction method for time series regression based on data bootstrapping.