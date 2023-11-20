import unittest

from jax import random
import numpy as np
import torch

from fortuna.data import InputsLoader
from fortuna.hallucination.scoring.inv_perplexity import inv_perplexity


class TestScoringModel(unittest.TestCase):
    def test_score(self):
        logits = torch.ones((5, 10, 3))
        labels = torch.ones(
            (
                4,
                10,
            )
        )

        assert inv_perplexity(logits=logits, labels=labels).shape == ()
        assert inv_perplexity(logits=logits, labels=labels, init_pos=2).shape == ()
