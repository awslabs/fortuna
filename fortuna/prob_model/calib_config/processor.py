class CalibProcessor:
    def __init__(
        self, gpus: int = -1, disable_jit: bool = False, n_posterior_samples: int = 30
    ):
        """
        An object to configure computational aspects of the calibration process.

        Parameters
        ----------
        gpus: int
            A list of devices to be used during training.
            At the moment two options are supported: use all devices (`gpus=-1`) or use no device (`gpus=0`).
        disable_jit: bool
            if True, no function within the calibration loop is jitted.
        n_posterior_samples: int
            Number of posterior samples to draw from the posterior distribution for the calibration process.
        """
        self.gpus = gpus
        self.disable_jit = disable_jit
        self.n_posterior_samples = n_posterior_samples
