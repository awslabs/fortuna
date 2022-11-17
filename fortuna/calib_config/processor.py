class CalibProcessor:
    def __init__(
        self, n_posterior_samples: int = 30,
    ):
        """
        An object to configure computational aspects of the calibration process.

        :param n_posterior_samples: int = 30
            Number of posterior samples.
        """
        self.n_posterior_samples = n_posterior_samples
