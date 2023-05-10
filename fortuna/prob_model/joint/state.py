from __future__ import annotations

from typing import Optional

from fortuna.model.model_manager.state import ModelManagerState
from fortuna.output_calibrator.output_calib_manager.state import OutputCalibManagerState
from fortuna.typing import CalibMutable, CalibParams, Mutable, Params


class JointState(ModelManagerState):
    params: Params
    mutable: Optional[Mutable] = None
    calib_params: Optional[CalibParams] = None
    calib_mutable: Optional[CalibMutable] = None

    def __init__(
        self,
        params: Params,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
    ):
        """
        A joint distribution state. This includes all the parameters and mutable objects of the joint distribution.

        Parameters
        ----------
        params : Params
            The random parameters of the probabilistic model.
        mutable : Optional[Mutable]
            The mutable objects used to evaluate the models.
        calib_params : Optional[CalibParams]
            The calibration parameters of the probabilistic model.
        calib_mutable : Optional[CalibMutable]
            The calibration mutable objects used to evaluate the calibrators.
        """
        super(JointState, self).__init__(params, mutable)
        self.calib_params = calib_params
        self.calib_mutable = calib_mutable

    @classmethod
    def init_from_states(
        cls,
        model_manager_state: ModelManagerState,
        output_calib_manager_state: Optional[OutputCalibManagerState] = None,
    ) -> JointState:
        """
        Initialize a probabilistic model state from an model manager state and a calibration state.

        Parameters
        ----------
        model_manager_state: ModelManagerState
            An model manager state.
        output_calib_manager_state: OutputCalibManagerState
            An output calibration manager state.

        Returns
        -------
        JointState
            A joint distribution state.
        """
        return cls(
            params=model_manager_state.params,
            mutable=model_manager_state.mutable,
            calib_params=getattr(output_calib_manager_state, "params", None),
            calib_mutable=getattr(output_calib_manager_state, "mutable", None),
        )
