from typing import Optional


class Hyperparameters:
    def __init__(
        self,
        max_grad_norm: Optional[float] = None,
        gradient_accumulation_steps: Optional[int] = None,
    ):
        """
        An object to configure additional arguments that may be needed during the posterior fitting.

        Parameters
        ----------
        max_grad_norm: Optional[Path]
            Maximum gradient norm. If `max_grad_norm > 0`, gradient clipping is performed.
        gradient_accumulation_steps: Optional[Path]
            Number of forward passes to perform before doing a backward pass.
        """
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
