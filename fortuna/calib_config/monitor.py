class CalibMonitor:
    def __init__(
        self, verbose: bool = True,
    ):
        """
        An object to configure the monitoring of the calibration process.

        Parameters
        ----------
        verbose: bool
            Whether to log the training progress.
        """
        self.verbose = verbose
