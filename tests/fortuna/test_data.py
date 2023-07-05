import unittest

import numpy as np

from fortuna.data.loader import (
    DataLoader,
    InputsLoader,
    TargetsLoader,
)
from tests.make_data import (
    make_array_random_data,
    make_generator_fun_random_data,
)


class Test(unittest.TestCase):
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
    def test_data_loader_from_inputs_loaders(self):
        inputs1 = np.arange(10)
        inputs2 = 1 + np.arange(7)
        inputs_loaders = [
            InputsLoader.from_array_inputs(inputs1, batch_size=3),
            InputsLoader.from_array_inputs(inputs2, batch_size=4),
        ]
        targets = [0, 1]
        data_loader = DataLoader.from_inputs_loaders(
            inputs_loaders, targets, how="interpose"
        )
        for i, (x, y) in enumerate(data_loader):
            assert (
                x.shape == (7,)
                if i == 0
                else (6,)
                if i == 1
                else (3,)
                if i == 2
                else (1,)
            )
            assert all(y[:3] == 0)
            assert all(y[3:] == 1)
            assert len(x) == len(y)

        data_loader = DataLoader.from_inputs_loaders(
            inputs_loaders, targets, how="interpose"
        )
        for i, (x, y) in enumerate(data_loader):
            assert (
                x.shape == (3,)
                if i == 0
                else (3,)
                if i == 1
                else (3,)
                if i == 2
                else (1,)
                if i == 3
                else (4,)
                if i == 4
                else (3,)
            )
            if i <= 3:
                assert all(y == 0)
            elif i > 3:
                assert all(y == 1)
            assert len(x) == len(y)

    def test_inputs_loader_to_transformed_inputs_loader(self):
        inputs = np.arange(10)
        inputs_loader = InputsLoader.from_array_inputs(inputs, batch_size=3)
        transformed_inputs_loader = inputs_loader.to_transformed_inputs_loader(
            lambda x, s: (x[x < 7], s)
        )

        for i, x in enumerate(transformed_inputs_loader):
            assert x.shape == (3,) if i < 2 else (1,)
            assert all(x < 7)

    def test_data_loader_to_transformed_data_loader(self):
        data = np.arange(10), np.arange(10)
        data_loader = DataLoader.from_array_data(data, batch_size=3)

        def transform(x, y, status):
            idx = x < 7
            status["k"] = "v"
            return x[idx], y[idx], status

        transformed_data_loader = data_loader.to_transformed_data_loader(
            transform, status={}
        )

        for i, (x, y) in enumerate(transformed_data_loader):
            assert x.shape == (3,) if i < 2 else (1,)
            assert all(x < 7)
            assert x.shape[0] == y.shape[0]

    def test_targets_loader_to_transformed_targets_loader(self):
        targets = np.arange(10)
        targets_loader = TargetsLoader.from_array_targets(targets, batch_size=3)
        transformed_targets_loader = targets_loader.to_transformed_targets_loader(
            lambda y, s: (y[y < 7], s)
        )

        for i, y in enumerate(transformed_targets_loader):
            assert y.shape == (3,) if i < 2 else (1,)
            assert all(y < 7)

    def test_sample_inputs_loader(self):
        inputs = np.arange(10)
        inputs_loader = InputsLoader.from_array_inputs(inputs, batch_size=3)
        sampled_loader = inputs_loader.sample(0, 6)
        assert len(sampled_loader.to_array_inputs()) == 6
        inputs_loader.sample(0, 16)

    def test_sample_data_loader(self):
        inputs, targets = np.arange(10), np.arange(10)
        data_loader = DataLoader.from_array_data((inputs, targets), batch_size=3)
        sampled_loader = data_loader.sample(0, 6)
        sampled = sampled_loader.to_array_data()
        assert len(sampled[0]) == 6
        assert len(sampled[1]) == 6
        data_loader.sample(0, 16)

    def test_sample_targets_loader(self):
        targets = np.arange(10)
        targets_loader = TargetsLoader.from_array_targets(targets, batch_size=3)
        sampled_loader = targets_loader.sample(0, 6)
        assert len(sampled_loader.to_array_targets()) == 6
        targets_loader.sample(0, 16)

    def test_split_data_loader(self):
        inputs, targets = np.arange(10), 1 + np.arange(10)
        data_loader = DataLoader.from_array_data((inputs, targets), batch_size=4)
        data_loader1, data_loader2 = data_loader.split(7)
        c = 0
        for x, y in data_loader1:
            c += x.shape[0]
            assert all(x <= 6)
            assert all(y <= 7)
            assert len(x) == len(y)
        assert c == 7
        c = 0
        for x, y in data_loader2:
            c += x.shape[0]
            assert all(x > 6)
            assert all(y > 7)
            assert len(x) == len(y)
        assert c == 3

    def test_split_inputs_loader(self):
        inputs = np.arange(10)
        inputs_loader = InputsLoader.from_array_inputs(inputs, batch_size=4)
        inputs_loader1, inputs_loader2 = inputs_loader.split(7)
        c = 0
        for x in inputs_loader1:
            c += x.shape[0]
            assert all(x <= 6)
        assert c == 7
        c = 0
        for x in inputs_loader2:
            c += x.shape[0]
            assert all(x > 6)
        assert c == 3

    def test_split_targets_loader(self):
        targets = np.arange(10)
        targets_loader = TargetsLoader.from_array_targets(targets, batch_size=4)
        targets_loader1, targets_loader2 = targets_loader.split(7)
        c = 0
        for y in targets_loader1:
            c += y.shape[0]
            assert all(y <= 6)
        assert c == 7
        c = 0
        for y in targets_loader2:
            c += y.shape[0]
            assert all(y > 6)
        assert c == 3

    def test_size_data_loader(self):
        data = np.arange(10), np.arange(10)
        data_loader = DataLoader.from_array_data(data, batch_size=3)
        assert data_loader.size == 10

    def test_size_inputs_loader(self):
        inputs = np.arange(10)
        inputs_loader = InputsLoader.from_array_inputs(inputs, batch_size=3)
        assert inputs_loader.size == 10

    def test_size_targets_loader(self):
        targets = np.arange(10)
        targets_loader = TargetsLoader.from_array_targets(targets, batch_size=3)
        assert targets_loader.size == 10

    def test_chop_data_loader(self):
        data = np.arange(10), np.arange(10)
        data_loader = DataLoader.from_array_data(data, batch_size=3).chop(2)
        for x, y in data_loader:
            assert x.shape[0] == 2
            assert y.shape[0] == 2

    def test_chop_inputs_loader(self):
        inputs = np.arange(10)
        inputs_loader = InputsLoader.from_array_inputs(inputs, batch_size=3).chop(2)
        for x in inputs_loader:
            assert x.shape[0] == 2

    def test_chop_targets_loader(self):
        targets = np.arange(10)
        targets_loader = TargetsLoader.from_array_targets(targets, batch_size=3).chop(2)
        for y in targets_loader:
            assert y.shape[0] == 2
