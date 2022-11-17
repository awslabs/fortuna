import unittest

import numpy as np
from fortuna.data.loader import DataLoader, InputsLoader, TargetsLoader
from tests.make_data import (make_generator_fun_random_data,
                             make_generator_random_data)


class Test(unittest.TestCase):
    def test_get_inputs_from_generator(self):
        data = make_generator_random_data(
            n_batches=2,
            batch_size=3,
            shape_inputs=(5,),
            output_dim=4,
            output_type="continuous",
        )
        data = DataLoader.from_iterable(data)
        inputs = InputsLoader.from_data_loader(data)
        for a in inputs:
            assert type(a) == np.ndarray
            assert a.shape == (3, 5)

    def test_get_inputs_from_generator_function(self):
        data = make_generator_fun_random_data(
            n_batches=2,
            batch_size=3,
            shape_inputs=(5,),
            output_dim=4,
            output_type="continuous",
        )
        data = DataLoader.from_callable_iterable(data)
        inputs = InputsLoader.from_data_loader(data)
        for a in inputs:
            assert type(a) == np.ndarray
            assert a.shape == (3, 5)


class TestGetTargets(unittest.TestCase):
    def test_get_targets_from_generator(self):
        data = make_generator_random_data(
            n_batches=2,
            batch_size=3,
            shape_inputs=(5,),
            output_dim=4,
            output_type="continuous",
        )
        data = DataLoader.from_iterable(data)
        targets = TargetsLoader.from_data_loader(data)
        for a in targets:
            assert type(a) == np.ndarray
            assert a.shape == (3, 4)

    def test_get_targets_from_generator_function(self):
        data = make_generator_fun_random_data(
            n_batches=2,
            batch_size=3,
            shape_inputs=(5,),
            output_dim=4,
            output_type="continuous",
        )
        data = DataLoader.from_callable_iterable(data)
        targets = TargetsLoader.from_data_loader(data)
        for a in targets:
            assert type(a) == np.ndarray
            assert a.shape == (3, 4)
