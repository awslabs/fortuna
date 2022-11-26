import unittest

import numpy as np

from fortuna.metric.classification import brier_score


class TestBrierScore(unittest.TestCase):
    def test_brier_score(self):
        probs = np.random.normal(size=(10, 3))
        targets = np.random.normal(size=10)
        assert np.atleast_1d(brier_score(probs, targets)).shape == (1,)
