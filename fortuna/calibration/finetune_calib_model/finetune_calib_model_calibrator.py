from fortuna.training.trainer import TrainerABC, JittedMixin, MultiDeviceMixin


class FinetuneCalibModelCalibrator(TrainerABC):
    pass


class JittedFinetuneCalibModelCalibrator(JittedMixin, FinetuneCalibModelCalibrator):
    pass


class MultiDeviceFinetuneCalibModelCalibrator(MultiDeviceMixin, FinetuneCalibModelCalibrator):
    pass