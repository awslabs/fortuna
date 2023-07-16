import enum

from fortuna.calib_model.state import CalibState


class NameToCalibState(enum.Enum):
    vars()[CalibState.__name__] = CalibState
