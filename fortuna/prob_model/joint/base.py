from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

from flax.core import FrozenDict
from jax import random
from jax._src.prng import PRNGKeyArray
import jax.numpy as jnp

from fortuna.likelihood.base import Likelihood
from fortuna.model.model_manager.state import ModelManagerState
from fortuna.output_calibrator.output_calib_manager.state import OutputCalibManagerState
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.prior.base import Prior
from fortuna.typing import (
    Batch,
    CalibMutable,
    CalibParams,
    Mutable,
    Params,
    Shape,
)
from fortuna.utils.data import get_inputs_from_shape
from fortuna.utils.random import WithRNG


class Joint(WithRNG):
    def __init__(self, prior: Prior, likelihood: Likelihood):
        r"""
        Joint distribution class. This is the joint distribution of target variables and random model parameters given
        input variables and calibration parameters. It is given by

        .. math::
            p(y, w|x, \phi),

        where:
         - :math:`x` is an observed input variable;
         - :math:`y` is an observed target variable;
         - :math:`w` denotes the random model parameters;
         - :math:`\phi` denotes the calibration parameters.

        Parameters
        ----------
        prior : Prior
            A prior distribution.
        likelihood : Likelihood
            A likelihood function.
        """
        self.prior = prior
        self.likelihood = likelihood

    def _batched_log_joint_prob(
        self,
        params: Params,
        batch: Batch,
        n_data: int,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        return_aux: Optional[List[str]] = None,
        train: Optional[bool] = False,
        outputs: Optional[jnp.ndarray] = None,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs,
    ) -> Union[float, Tuple[float, dict]]:
        """
        Evaluate the joint batched log-probability density function (a.k.a. log-pdf).

        Parameters
        ----------
        params : Params
            The random parameters of the probabilistic model.
        batch : Batch
            A batch of data points.
        n_data : int
            The total number of data points over which the likelihood is joint. This is used to rescale the batched
            log-likelihood function to better approximate the full likelihood.
        mutable : Optional[Mutable]
            The mutable objects used to evaluate the models.
        calib_params : Optional[CalibParams]
            The calibration parameters of the probabilistic model.
        calib_mutable : Optional[CalibMutable]
            The calibration mutable objects used to evaluate the calibrators.
        return_aux : Optional[List[str]]
            The auxiliary objects to return. We support 'outputs' and 'calib_mutable'. If this argument is not given,
            no auxiliary object is returned.
        train : Optional[bool]
            Whether the method is called during training.
        outputs : Optional[jnp.ndarray]
            Pre-computed batch of outputs.
        rng: Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        Union[float, Tuple[float, dict]]
            The evaluation of the joint batched log-pdf. If `return_aux` is given, the corresponding auxiliary objects
            are also returned.
        """
        if return_aux is None:
            return_aux = []
        log_prior = self.prior.log_joint_prob(params)
        outs = self.likelihood._batched_log_joint_prob(
            params,
            batch,
            n_data,
            mutable=mutable,
            calib_params=calib_params,
            calib_mutable=calib_mutable,
            return_aux=return_aux,
            train=train,
            outputs=outputs,
            rng=rng,
            **kwargs,
        )
        if len(return_aux) > 0:
            batched_log_lik, aux = outs
            return batched_log_lik + log_prior, aux
        batched_log_lik = outs
        return batched_log_lik + log_prior

    def _batched_negative_log_joint_prob(
        self,
        params: Params,
        batch: Batch,
        n_data: int,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        return_aux: Optional[List[str]] = None,
        train: Optional[bool] = False,
        outputs: Optional[jnp.ndarray] = None,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs,
    ) -> Union[float, Tuple[float, dict]]:
        outs = self._batched_log_joint_prob(
            params,
            batch,
            n_data,
            mutable,
            calib_params,
            calib_mutable,
            return_aux,
            train,
            outputs,
            rng,
            **kwargs,
        )
        if len(return_aux) > 0:
            loss, aux = outs
            loss *= -1
            return loss, aux
        return -outs

    def init(
        self, input_shape: Shape, rng: Optional[PRNGKeyArray] = None, **kwargs
    ) -> JointState:
        """
        Initialize the state of the joint distribution.

        Parameters
        ----------
        input_shape : Shape
            The shape of the input variable.
        rng: Optional[PRNGKeyArray]
            A random number generator key.

        Returns
        -------
        A state of the joint distribution.
        """
        if rng is None:
            rng = self.rng.get()
        key1, key2 = random.split(rng)

        oms = ModelManagerState.init_from_dict(
            self.likelihood.model_manager.init(input_shape, rng=key1, **kwargs)
        )
        inputs = get_inputs_from_shape(input_shape)
        outputs = self.likelihood.model_manager.apply(
            params=oms.params, inputs=inputs, mutable=oms.mutable
        )
        output_dim = (
            outputs[0].shape[-1]
            if isinstance(outputs, (list, tuple))
            else outputs.shape[-1]
        )
        ocms = OutputCalibManagerState.init_from_dict(
            FrozenDict(
                output_calibrator=self.likelihood.output_calib_manager.init(
                    output_dim=output_dim, rng=key2
                )
            )
        )
        return JointState.init_from_states(oms, ocms)
