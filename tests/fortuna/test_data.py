import unittest

import numpy as np

from fortuna.data.loader import DataLoader, InputsLoader, TargetsLoader
from tests.make_data import (make_array_random_data,
                             make_generator_fun_random_data,
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

    def test_array_data(self):
        data_org = make_array_random_data(
            n_data=10,
            shape_inputs=(2,),
            output_dim=1,
            output_type="continuous",
        )
        for prefetch in [False, True]:
            data_new = DataLoader.from_array_data(
                data_org, batch_size=2, prefetch=prefetch
            ).to_array_data()
            for original, new in zip(data_org, data_new):
                assert (original == new).all()


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


class TestDataLoaders(unittest.TestCase):
    def test_data_loader_from_inputs_loader(self):
        inputs1 = np.arange(10)
        inputs2 = 1 + np.arange(7)
        inputs_loaders = [
            InputsLoader.from_array_inputs(inputs1, batch_size=3),
            InputsLoader.from_array_inputs(inputs2, batch_size=4)
        ]
        targets = [0, 1]
        data_loader = DataLoader.from_inputs_loader(inputs_loaders, targets)
        for i, (x, y) in enumerate(data_loader):
            assert x.shape == (7,) if i == 0 else (6,) if i == 1 else (3,) if i == 2 else (1,)
            assert all(y[:3] == 0)
            assert all(y[3:] == 1)
            assert len(x) == len(y)

    def test_inputs_loader_to_filtered_inputs_loader(self):
        inputs = np.arange(10)
        inputs_loader = InputsLoader.from_array_inputs(inputs, batch_size=3)
        filtered_inputs_loader = inputs_loader.to_filtered_inputs_loader(lambda x: x[x < 7])

        for i, x in enumerate(filtered_inputs_loader):
            assert x.shape == (3,) if i < 2 else (1,)
            assert all(x < 7)
