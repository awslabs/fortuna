import abc
from typing import Optional

import jax

from fortuna.typing import Params
from fortuna.utils.random import WithRNG


class Prior(WithRNG, abc.ABC):
    """
    Abstract prior distribution class.
    """

    @abc.abstractmethod
    def log_joint_prob(self, params: Params) -> float:
        """
        Evaluate the prior log-probability density function (a.k.a. log-pdf).

        Parameters
        ----------
        params : PyTree
            The parameters where to evaluate the log-pdf.

        Returns
        -------
        float
            Evaluation of the prior log-pdf.
        """
        pass

    @abc.abstractmethod
    def sample(self, params_like: Params, rng: Optional[jax.Array] = None) -> Params:
        """
        Sample parameters from the prior distribution.

        Parameters
        ----------
        params_like : PyTree
            An PyTree object with the same structure as the parameters to sample.
        rng: Optional[jax.Array]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        PyTree
            A sample from the prior distribution.
        """
        pass
