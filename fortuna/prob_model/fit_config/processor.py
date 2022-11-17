class FitProcessor:
    def __init__(
        self, gpus: int = -1, disable_jit: bool = False,
    ):
        """
        An object to configure computational aspects of the calibration process

        Parameters
        ----------
        gpus: int
            A list of devices to be used during training.
            At the moment two options are supported: use all devices (`gpus=-1`) or use no device (`gpus=0`).
        disable_jit: bool
            if True, no function within the training loop is jitted.
        """
        self.gpus = gpus
        self.disable_jit = disable_jit
