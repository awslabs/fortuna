from __future__ import annotations

from copy import deepcopy
from itertools import zip_longest
from typing import (
    Iterable,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from fortuna.typing import (
    Array,
    Batch,
    InputData,
    Targets,
)


class IterableData:
    def __init__(
        self,
        generator,
    ):
        self._g = generator

    def __iter__(self) -> Iterable[Union[Batch, InputData, Targets]]:
        yield from iter(self._g())

    @classmethod
    def from_iterable(cls, iterable) -> IterableData:
        def _inner():
            yield from iterable

        return cls(_inner)

    @classmethod
    def from_callable(cls, fn) -> IterableData:
        def _inner():
            yield from fn()

        return cls(_inner)

    @classmethod
    def from_tf_dataloader(cls, tf_dataloader) -> IterableData:
        def _inner():
            for batch_inputs, batch_targets in tf_dataloader:
                if not isinstance(batch_inputs, dict):
                    batch_inputs = batch_inputs.numpy()
                else:
                    batch_inputs = {k: v.numpy() for k, v in batch_inputs.items()}
                batch_targets = batch_targets.numpy()
                yield batch_inputs, batch_targets

        return cls(_inner)

    @classmethod
    def from_torch_dataloader(cls, torch_dataloader) -> IterableData:
        def _inner():
            for batch_inputs, batch_targets in torch_dataloader:
                if not isinstance(batch_inputs, dict):
                    batch_inputs = batch_inputs.numpy()
                else:
                    batch_inputs = {k: v.numpy() for k, v in batch_inputs.items()}
                batch_targets = batch_targets.numpy()
                yield batch_inputs, batch_targets

        return cls(_inner)

    @classmethod
    def from_array_data(
        cls,
        data: Array,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        prefetch: bool = False,
        seed: Optional[int] = 0,
    ):
        def _inner():
            if shuffle:
                rng = np.random.default_rng(seed)
                perm = rng.choice(data.shape[0], data.shape[0], replace=False)
            if batch_size is None:
                yield data
            else:
                batches = np.split(
                    data[perm] if shuffle else data,
                    np.arange(batch_size, data.shape[0], batch_size),
                    axis=0,
                )

                def make_gen():
                    for batch in batches:
                        yield batch

                yield from PrefetchedGenerator(make_gen()) if prefetch else make_gen()

        return cls(_inner)

    @classmethod
    def from_batch_array_data(
        cls,
        data: Tuple[Array, Array],
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        prefetch: bool = False,
        seed: Optional[int] = 0,
    ):
        def _inner():
            if shuffle:
                rng = np.random.default_rng(seed)
                perm = rng.choice(data[0].shape[0], data[0].shape[0], replace=False)
            if batch_size is None:
                yield data
            else:
                x_batches = np.split(
                    data[0][perm] if shuffle else data[0],
                    np.arange(batch_size, data[0].shape[0], batch_size),
                    axis=0,
                )
                y_batches = np.split(
                    data[1][perm] if shuffle else data[1],
                    np.arange(batch_size, data[1].shape[0], batch_size),
                    axis=0,
                )

                def make_gen():
                    for x_batch, y_batch in zip(x_batches, y_batches):
                        yield x_batch, y_batch

                yield from PrefetchedGenerator(make_gen()) if prefetch else make_gen()

        return cls(_inner)

    @classmethod
    def data_loader_to_inputs_iterable(cls, data_loader) -> IterableData:
        def _inner():
            for inputs, _ in data_loader:
                yield inputs

        return cls(_inner)

    @classmethod
    def data_loader_to_targets_iterable(cls, data_loader) -> IterableData:
        def _inner():
            for _, targets in data_loader:
                yield targets

        return cls(_inner)

    @classmethod
    def inputs_loaders_to_batch_iterable(
        cls, inputs_loaders, targets, how
    ) -> IterableData:
        if how == "interpose":

            def _inner():
                for all_inputs in zip_longest(*inputs_loaders):
                    _targets = [
                        target * np.ones(inputs.shape[0], dtype="int32")
                        for inputs, target in zip(all_inputs, targets)
                        if inputs is not None
                    ]
                    all_inputs = [inputs for inputs in all_inputs if inputs is not None]
                    yield np.concatenate(all_inputs), np.concatenate(_targets)

        elif how == "concatenate":

            def _inner():
                for inputs_loader, target in zip(inputs_loaders, targets):
                    for inputs in inputs_loader:
                        yield inputs, target * np.ones(inputs.shape[0], dtype="int32")

        else:
            hows = ["interpose", "concatenate"]
            raise ValueError(
                f"`how={how} not recognized. Please choose among the following options: {hows}."
            )
        return cls(_inner)

    @classmethod
    def transform_data_loader(cls, loader, transform_fn, status) -> IterableData:
        def _inner():
            _status = deepcopy(status)
            for inputs, targets in loader:
                inputs, targets, _status = transform_fn(inputs, targets, _status)
                if (
                    inputs is not None
                    and targets is not None
                    and len(inputs) > 0
                    and len(targets) > 0
                ):
                    if inputs.shape[0] != targets.shape[0]:
                        raise ValueError(
                            "The first dimension of transformed inputs and targets must be the same, but "
                            f"{inputs.shape[0]} and {targets.shape[0]} were found."
                        )
                    yield inputs, targets

        return cls(_inner)

    @classmethod
    def transform_inputs_or_targets_loader(
        cls, loader, transform_fn, status
    ) -> IterableData:
        def _inner():
            _status = deepcopy(status)
            for data in loader:
                data, _status = transform_fn(data, _status)
                if data is not None and len(data) > 0:
                    yield data

        return cls(_inner)


class PrefetchedGenerator:
    def __init__(self, generator):
        self._batch = generator.__next__()
        self._generator = generator
        self._ready = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self._ready:
            self._prefetch()
        self._ready = False
        return self._batch

    def _prefetch(self):
        if not self._ready:
            self._batch = self._generator.__next__()
            self._ready = True
