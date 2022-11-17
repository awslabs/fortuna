import os
import tempfile
import unittest

import numpy as np
from fortuna.plot import plot_reliability_diagram


class TestStates(unittest.TestCase):
    def test_reliability_diagram(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            accs = [np.random.normal(size=20), np.random.normal(size=20)]
            confs = [np.random.normal(size=20), np.random.normal(size=20)]
            labels = ["a", "b"]
            plot_reliability_diagram(accs, confs)
            plot_reliability_diagram(accs[0], confs[0])
            plot_reliability_diagram(accs, confs, labels=labels)
            plot_reliability_diagram(
                accs, confs, fname=os.path.join(tmp_dir, "tmp.png")
            )
            plot_reliability_diagram(
                accs, confs, fname=os.path.join(tmp_dir, "tmp.png")
            )
            plot_reliability_diagram(accs, confs, title="bla")
