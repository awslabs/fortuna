class CalibProcessor:
    def __init__(
        self, devices: int = -1, disable_jit: bool = False,
    ):
        """
        An object to configure computational aspects of the calibration process.

        Parameters
        ----------
        devices: int
            A list of devices to be used during training.
            At the moment two options are supported: use all devices (`devices=-1`) or use no device (`devices=0`).
        disable_jit: bool
            if True, no function within the calibration loop is jitted.
        """
        self.devices = devices
        self.disable_jit = disable_jit
