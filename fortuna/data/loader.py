from __future__ import annotations

from typing import Callable, Iterable, Optional, Union

import jax
import numpy as np
from flax import jax_utils
from jax.tree_util import tree_map

from fortuna.typing import Array, Batch


class DataLoader:
    def __init__(
        self,
        data_loader: Union[
            FromIterableToDataLoader,
            FromCallableIterableToDataLoader,
            FromArrayDataToDataLoader,
            FromTensorFlowDataLoaderToDataLoader,
            FromTorchDataLoaderToDataLoader,
            ChoppedDataLoader
        ],
    ):
        """
        A data loader class. Each batch is a tuple of input and target variables, respectively. Both inputs and targets
        are arrays, with different data points over the first dimension.

        Parameters
        ----------
        data_loader : Union[FromIterableToDataLoader, FromCallableIterableToDataLoader, FromArrayDataToDataLoader,
        FromTensorFlowDataLoaderToDataLoader, FromTorchDataLoaderToDataLoader]
            A data loader.
        """
        self._data_loader = data_loader

    def __iter__(self):
        yield from self._data_loader()

    @classmethod
    def from_array_data(
        cls,
        data: Batch,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        prefetch: bool = False,
    ) -> DataLoader:
        """
        Build a :class:`~fortuna.data.loader.DataLoader` object from a tuple of arrays of input and target variables,
        respectively.

        Parameters
        ----------
        data: Batch
            Input and target arrays of data.
        batch_size: Optional[int]
            The batch size. If not given, the data will not be batched.
        shuffle: bool
            Whether the data loader should shuffle at every call.
        prefetch: bool
            Whether to prefetch the next batch.

        Returns
        -------
        DataLoader
            A data loader built out of the tuple of arrays.
        """
        return cls(
            data_loader=FromArrayDataToDataLoader(
                data, batch_size=batch_size, shuffle=shuffle, prefetch=prefetch
            )
        )

    @classmethod
    def from_callable_iterable(cls, fun: Callable[[], Iterable[Batch],],) -> DataLoader:
        """
        Transform a callable iterable into a :class:`~fortuna.data.loader.DataLoader` object.

        Parameters
        ----------
        fun: Callable[[], Iterable[Batch]]
            A callable iterable of tuples of input and target arrays.

        Returns
        -------
        DataLoader
            A data loader object.
        """
        return cls(data_loader=FromCallableIterableToDataLoader(fun))

    @classmethod
    def from_iterable(cls, iterable: Iterable[Batch],) -> DataLoader:
        """
        Transform an iterable into a :class:`~fortuna.data.loader.DataLoader` object.

        Parameters
        ----------
        iterable: Iterable[Batch]
            An iterable of tuples of input and target arrays.

        Returns
        -------
        DataLoader
            A data loader object.
        """
        return cls(data_loader=FromIterableToDataLoader(iterable))

    @classmethod
    def from_tensorflow_data_loader(cls, tf_data_loader) -> DataLoader:
        """
        Transform a TensorFlow data loader into a :class:`~fortuna.data.loader.DataLoader` object.

        Parameters
        ----------
        tf_data_loader
            A TensorFlow data loader where each batch is a tuple of input and target Tensors.

        Returns
        -------
        DataLoader
            A data loader object.
        """
        return cls(
            data_loader=FromTensorFlowDataLoaderToDataLoader(
                tf_data_loader=tf_data_loader
            )
        )

    @classmethod
    def from_torch_data_loader(cls, torch_data_loader) -> DataLoader:
        """
        Transform a PyTorch data loader into a :class:`~fortuna.data.loader.DataLoader` object.

        Parameters
        ----------
        torch_data_loader
            A PyTorch data loader where each batch is a tuple of input and target Tensors.

        Returns
        -------
        DataLoader
            A data loader object.
        """
        return cls(
            data_loader=FromTorchDataLoaderToDataLoader(
                torch_data_loader=torch_data_loader
            )
        )

    def to_array_data(self) -> Batch:
        """
        Reduce a data loader to a tuple of input and target arrays.

        Returns
        -------
        Batch
            Tuple of input and target arrays.
        """
        inputs, targets = [], []
        for batch_inputs, batch_targets in self._data_loader():
            inputs.append(batch_inputs)
            targets.append(batch_targets)
        return np.concatenate(inputs, 0), np.concatenate(targets, 0)

    def to_array_inputs(self) -> Array:
        """
        Reduce a data loader to an array of target data.

        Returns
        -------
        Array
            Array of input data.
        """
        inputs = []
        for batch_inputs, batch_targets in self._data_loader():
            inputs.append(batch_inputs)
        return np.concatenate(inputs, 0)

    def to_array_targets(self) -> Array:
        """
        Reduce a data loader to an array of target data.

        Returns
        -------
        Array
            Array of input data.
        """
        targets = []
        for batch_inputs, batch_targets in self._data_loader():
            targets.append(batch_targets)
        return np.concatenate(targets, 0)

    def to_inputs_loader(self) -> InputsLoader:
        """
        Reduce a data loader to an inputs loader.

        Returns
        -------
        InputsLoader
            The inputs loader derived from the data loader.
        """
        return InputsLoader.from_data_loader(DataLoader(data_loader=self._data_loader))

    def to_targets_loader(self) -> TargetsLoader:
        """
        Reduce a data loader to a targets loader.

        Returns
        -------
        TargetsLoader
            The targets loader derived from the data loader.
        """
        return TargetsLoader.from_data_loader(DataLoader(data_loader=self._data_loader))

    @classmethod
    def chop(cls, data_loader: DataLoader, divisor: int) -> DataLoader:
        """
        Chop the last part of each batch of the data loader, to make sure the number od data points per batch divides
        `divisor`.

        Parameters
        ----------
        data_loader : DataLoader
            A data loader
        divisor : int
            Number of data points that each batched must divide.

        Returns
        -------
        DataLoader
            A data loader with chopped batches.
        """
        return cls(data_loader=ChoppedDataLoader(data_loader=data_loader, divisor=divisor))


class InputsLoader:
    def __init__(
        self,
        inputs_loader: Union[
            FromArrayInputsToInputsLoader,
            FromDataLoaderToInputsLoader,
            FromCallableIterableToInputsLoader,
            FromIterableToInputsLoader,
            ChoppedInputsLoader
        ],
    ):
        """
        An inputs loader class. Each batch is an array of inputs, with different data points over the first dimension.

        Parameters
        ----------
        inputs_loader : Union[FromArrayInputsToInputsLoader, FromDataLoaderToInputsLoader]
            An inputs loader.
        """
        self._inputs_loader = inputs_loader

    def __iter__(self):
        yield from self._inputs_loader()

    @classmethod
    def from_data_loader(cls, data_loader: DataLoader) -> InputsLoader:
        """
        Reduce a data loader to an inputs loader.

        Parameters
        ----------
        data_loader : DataLoader
            A data loader.

        Returns
        -------
        InputsLoader
            An inputs loader.
        """
        return cls(inputs_loader=FromDataLoaderToInputsLoader(data_loader))

    @classmethod
    def from_array_inputs(
        cls,
        inputs: Array,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        prefetch: bool = False,
    ) -> InputsLoader:
        """
        Build a :class:`~fortuna.data.loader.InputsLoader` object from an array of input data.

        Parameters
        ----------
        inputs: Array
            Input array of data.
        batch_size: Optional[int]
            The batch size. If not given, the inputs will not be batched.
        shuffle: bool
            Whether the inputs loader should shuffle at every call.
        prefetch: bool
            Whether to prefetch the next batch.

        Returns
        -------
        InputsLoader
            An inputs loader built out of the array of inputs.
        """
        return cls(
            inputs_loader=FromArrayInputsToInputsLoader(
                inputs, batch_size=batch_size, shuffle=shuffle, prefetch=prefetch
            )
        )

    def to_array_inputs(self) -> Array:
        """
        Reduce an inputs loader to an array of inputs.

        Returns
        -------
        Array
            Array of input data.
        """
        inputs = []
        for batch_inputs in self._inputs_loader():
            inputs.append(batch_inputs)
        return np.concatenate(inputs, 0)

    @classmethod
    def from_callable_iterable(
        cls, fun: Callable[[], Iterable[Array]],
    ) -> InputsLoader:
        """
        Transform a callable iterable into a :class:`~fortuna.data.loader.InputsLoader` object.

        Parameters
        ----------
        fun: Callable[[], Iterable[Array]]
            A callable iterable of input arrays.

        Returns
        -------
        InputsLoader
            An inputs loader object.
        """
        return cls(inputs_loader=FromCallableIterableToInputsLoader(fun))

    @classmethod
    def from_iterable(cls, iterable: Iterable[Array],) -> InputsLoader:
        """
        Transform an iterable into a :class:`~fortuna.data.loader.InputsLoader` object.

        Parameters
        ----------
        iterable: Iterable[Array]
            An iterable of input arrays.

        Returns
        -------
        InputsLoader
            An inputs loader object.
        """
        return cls(inputs_loader=FromIterableToInputsLoader(iterable))

    @classmethod
    def chop(cls, inputs_loader: InputsLoader, divisor: int) -> InputsLoader:
        """
        Chop the last part of each batch of the inputs loader, to make sure the number od data points per batch divides
        `divisor`.

        Parameters
        ----------
        inputs_loader : InputsLoader
            An inputs loader.
        divisor : int
            Number of data points that each batched must divide.

        Returns
        -------
        InputsLoader
            An inputs loader with chopped batches.
        """
        return cls(inputs_loader=ChoppedInputsLoader(inputs_loader=inputs_loader, divisor=divisor))


class TargetsLoader:
    def __init__(
        self,
        targets_loader: Union[
            FromArrayTargetsToTargetsLoader,
            FromDataLoaderToTargetsLoader,
            FromCallableIterableToTargetsLoader,
            FromIterableToTargetsLoader,
            ChoppedTargetsLoader
        ],
    ):
        """
        A targets loader class. Each batch is an array of targets, with different data points over the first dimension.

        Parameters
        ----------
        targets_loader : Union[FromArrayTargetsToTargetsLoader, FromDataLoaderToTargetsLoader]
            A targets loader.
        """
        self._targets_loader = targets_loader

    def __iter__(self):
        yield from self._targets_loader()

    @classmethod
    def from_data_loader(cls, data_loader: DataLoader) -> TargetsLoader:
        """
        Reduce a data loader to a targets loader.

        Parameters
        ----------
        data_loader : DataLoader
            A data loader.

        Returns
        -------
        TargetsLoader
            A targets loader.
        """
        return cls(targets_loader=FromDataLoaderToTargetsLoader(data_loader))

    @classmethod
    def from_array_targets(
        cls,
        targets: Array,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        prefetch: bool = False,
    ) -> TargetsLoader:
        """
        Build a :class:`~fortuna.data.loader.TargetsLoader` object from an array of target data.

        Parameters
        ----------
        targets: Array
            Target array of data.
        batch_size: Optional[int]
            The batch size. If not given, the targets will not be batched.
        shuffle: bool
            Whether the target loader should shuffle at every call.
        prefetch: bool
            Whether to prefetch the next batch.

        Returns
        -------
        TargetsLoader
            A targets loader built out of the array of targets.
        """
        return cls(
            targets_loader=FromArrayTargetsToTargetsLoader(
                targets, batch_size=batch_size, shuffle=shuffle, prefetch=prefetch
            )
        )

    def to_array_targets(self) -> Array:
        """
        Reduce a targets loader to an array of targets.

        Returns
        -------
        Array
            Array of target data.
        """
        targets = []
        for batch_targets in self._targets_loader():
            targets.append(batch_targets)
        return np.concatenate(targets, 0)

    @classmethod
    def from_callable_iterable(
        cls, fun: Callable[[], Iterable[Array]],
    ) -> TargetsLoader:
        """
        Transform a callable iterable into a :class:`~fortuna.data.loader.TargetsLoader` object.

        Parameters
        ----------
        fun: Callable[[], Iterable[Array]]
            A callable iterable of target arrays.

        Returns
        -------
        TargetsLoader
            A targets loader object.
        """
        return cls(targets_loader=FromCallableIterableToTargetsLoader(fun))

    @classmethod
    def from_iterable(cls, iterable: Iterable[Array],) -> TargetsLoader:
        """
        Transform an iterable into a :class:`~fortuna.data.loader.TargetsLoader` object.

        Parameters
        ----------
        iterable: Iterable[Array]
            An iterable of target arrays.

        Returns
        -------
        TargetsLoader
            A targets loader object.
        """
        return cls(targets_loader=FromIterableToTargetsLoader(iterable))

    @classmethod
    def chop(cls, targets_loader: TargetsLoader, divisor: int) -> TargetsLoader:
        """
        Chop the last part of each batch of the targets loader, to make sure the number od data points per batch divides
        `divisor`.

        Parameters
        ----------
        targets_loader : TargetsLoader
            A targets loader.
        divisor : int
            Number of data points that each batched must divide.

        Returns
        -------
        InputsLoader
            A targets loader with chopped batches.
        """
        return cls(targets_loader=ChoppedTargetsLoader(targets_loader=targets_loader, divisor=divisor))


class FromDataLoaderToArrayData:
    def __init__(self, data_loader: DataLoader):
        self._data_loader = data_loader

    def __call__(self, *args, **kwargs):
        data = []
        for batch in self._data_loader:
            data.append(batch)
        return np.concatenate(data, 0)


class FromDataLoaderToInputsTargetsLoaders:
    def __init__(self, data_loader: DataLoader):
        self._data_loader = data_loader

    def __call__(self, *args, **kwargs):
        for x_batch, y_batch in self._data_loader:
            yield x_batch, y_batch


class FromArrayDataToDataLoader:
    def __init__(
        self,
        data: Batch,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        prefetch: bool = False,
    ):
        self._data = data
        self._batch_size = batch_size
        self._shuflle = shuffle
        self._prefetch = prefetch

    def __call__(self, *args, **kwargs):
        if self._shuflle:
            perm = np.random.choice(
                self._data[0].shape[0], self._data[0].shape[0], replace=False
            )
        if self._batch_size is None:
            yield self._data
        else:
            x_batches = np.split(
                self._data[0][perm] if self._shuflle else self._data[0],
                np.arange(self._batch_size, self._data[0].shape[0], self._batch_size),
                axis=0,
            )
            y_batches = np.split(
                self._data[1][perm] if self._shuflle else self._data[1],
                np.arange(self._batch_size, self._data[1].shape[0], self._batch_size),
                axis=0,
            )

            def make_gen():
                for x_batch, y_batch in zip(x_batches, y_batches):
                    yield x_batch, y_batch

            yield from PrefetchedGenerator(make_gen()) if self._prefetch else make_gen()


class FromCallableIterableToDataLoader:
    def __init__(
        self, fun: Callable[[], Iterable[Batch],],
    ):
        self._fun = fun

    def __call__(self, *args, **kwargs):
        return self._fun()


class FromCallableIterableToInputsLoader:
    def __init__(
        self, fun: Callable[[], Iterable[Array]],
    ):
        self._fun = fun

    def __call__(self, *args, **kwargs):
        return self._fun()


class FromCallableIterableToTargetsLoader:
    def __init__(
        self, fun: Callable[[], Iterable[Array]],
    ):
        self._fun = fun

    def __call__(self, *args, **kwargs):
        return self._fun()


class FromIterableToDataLoader:
    def __init__(
        self, batched_data: Iterable[Batch],
    ):
        self._batched_data = batched_data

    def __call__(self, *args, **kwargs):
        return self._batched_data


class FromIterableToInputsLoader:
    def __init__(
        self, batched_inputs: Iterable[Array],
    ):
        self._batched_inputs = batched_inputs

    def __call__(self, *args, **kwargs):
        return self._batched_inputs


class FromIterableToTargetsLoader:
    def __init__(
        self, batched_targets: Iterable[Array],
    ):
        self._batched_targets = batched_targets

    def __call__(self, *args, **kwargs):
        return self._batched_targets


class FromTensorFlowDataLoaderToDataLoader:
    def __init__(self, tf_data_loader):
        self._tf_data_loader = tf_data_loader

    def __call__(self, *args, **kwargs):
        for batch_inputs, batch_targets in self._tf_data_loader:
            yield batch_inputs.numpy(), batch_targets.numpy()


class FromTorchDataLoaderToDataLoader:
    def __init__(self, torch_data_loader):
        self._torch_data_loader = torch_data_loader

    def __call__(self, *args, **kwargs):
        for batch_inputs, batch_targets in self._torch_data_loader:
            yield batch_inputs.numpy(), batch_targets.numpy()


class FromDataLoaderToArrayInputs:
    def __init__(self, data_loader: DataLoader):
        self._data_loader = data_loader

    def __call__(self, *args, **kwargs):
        inputs = []
        for batch_inputs, batch_targets in self._data_loader:
            inputs.append(batch_inputs)
        return np.concatenate(inputs, 0)


class FromDataLoaderToArrayTargets:
    def __init__(self, data_loader: DataLoader):
        self._data_loader = data_loader

    def __call__(self, *args, **kwargs):
        targets = []
        for batch_inputs, batch_targets in self._data_loader:
            targets.append(batch_targets)
        return np.concatenate(targets, 0)


class FromInputsLoaderToArrayInputs:
    def __init__(self, inputs_loader: InputsLoader):
        self._inputs_loader = inputs_loader

    def __call__(self, *args, **kwargs):
        inputs = []
        for batch_inputs in self._inputs_loader:
            inputs.append(batch_inputs)
        return np.concatenate(inputs, 0)


class FromArrayInputsToInputsLoader:
    def __init__(
        self,
        inputs: Array,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        prefetch: bool = False,
    ):
        self._inputs = inputs
        self._batch_size = batch_size
        self._shuflle = shuffle
        self._prefetch = prefetch

    def __call__(self, *args, **kwargs):
        if self._shuflle:
            perm = np.random.choice(
                self._inputs.shape[0], self._inputs.shape[0], replace=False
            )
        if self._batch_size is None:
            yield self._inputs
        else:
            x_batches = np.split(
                self._inputs[perm] if self._shuflle else self._inputs,
                np.arange(self._batch_size, self._inputs.shape[0], self._batch_size),
                axis=0,
            )

            def make_gen():
                for x_batch in x_batches:
                    yield x_batch

            yield from PrefetchedGenerator(make_gen()) if self._prefetch else make_gen()


class FromDataLoaderToInputsLoader:
    def __init__(
        self, data_loader: DataLoader,
    ):
        self._data_loader = data_loader

    def __call__(self, *args, **kwargs):
        for inputs, targets in self._data_loader:
            yield inputs


class FromDataLoaderToTargetsLoader:
    def __init__(
        self, data_loader: DataLoader,
    ):
        self._data_loader = data_loader

    def __call__(self, *args, **kwargs):
        for inputs, targets in self._data_loader:
            yield targets


class FromTargetsLoaderToArrayTargets:
    def __init__(self, targets_loader: TargetsLoader):
        self._targets_loader = targets_loader

    def __call__(self, *args, **kwargs):
        targets = []
        for batch_targets in self._targets_loader:
            targets.append(batch_targets)
        return np.concatenate(targets, 0)


class FromArrayTargetsToTargetsLoader:
    def __init__(
        self,
        targets: Array,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        prefetch: bool = False,
    ):
        self._targets = targets
        self._batch_size = batch_size
        self._shuflle = shuffle
        self._prefetch = prefetch

    def __call__(self, *args, **kwargs):
        if self._shuflle:
            perm = np.random.choice(
                self._targets.shape[0], self._targets.shape[0], replace=False
            )
        if self._batch_size is None:
            yield self._targets
        else:
            x_batches = np.split(
                self._targets[perm] if self._shuflle else self._targets,
                np.arange(self._batch_size, self._targets.shape[0], self._batch_size),
                axis=0,
            )

            def make_gen():
                for x_batch in x_batches:
                    yield x_batch

            yield from PrefetchedGenerator(make_gen()) if self._prefetch else make_gen()


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


class ChoppedDataLoader:
    def __init__(self, data_loader: DataLoader, divisor: int):
        self._data_loader = data_loader
        self._divisor = divisor

    def __call__(self, *args, **kwargs):
        for inputs, targets in self._data_loader:
            reminder = targets.shape[0] % self._divisor
            if reminder == 0:
                yield inputs, targets
            else:
                yield inputs[:-reminder], targets[:-reminder]


class ChoppedInputsLoader:
    def __init__(self, inputs_loader: InputsLoader, divisor: int):
        self._inputs_loader = inputs_loader
        self._divisor = divisor

    def __call__(self, *args, **kwargs):
        for inputs in self._inputs_loader:
            reminder = inputs.shape[0] % self._divisor
            if reminder == 0:
                yield inputs
            else:
                yield inputs[:-reminder]


class ChoppedTargetsLoader:
    def __init__(self, targets_loader: TargetsLoader, divisor: int):
        self._targets_loader = targets_loader
        self._divisor = divisor

    def __call__(self, *args, **kwargs):
        for targets in self._targets_loader:
            reminder = targets.shape[0] % self._divisor
            if reminder == 0:
                yield targets
            else:
                yield targets[:-reminder]


class DeviceDimensionAugmentedDataLoader:
    def __init__(self, data_loader: DataLoader):
        self._data_loader = data_loader

    @staticmethod
    def _reshape_batch(batch):
        n_devices = jax.local_device_count()
        if batch.shape[0] % n_devices != 0:
            raise ValueError(
                f"The size of all batches must be a multiple of {n_devices}, that is the number of "
                f"available devices. However, a batch with shape {batch.shape[0]} was found. "
                f"Please set an appropriate batch size."
            )
        return batch.reshape((n_devices, -1) + batch.shape[1:])

    def __iter__(self, *args, **kwargs):
        data_loader = map(
            lambda batch: tree_map(self._reshape_batch, batch), self._data_loader
        )
        data_loader = jax_utils.prefetch_to_device(data_loader, 2)
        yield from data_loader


class DeviceDimensionAugmentedInputsLoader:
    def __init__(self, inputs_loader: InputsLoader):
        self._inputs_loader = inputs_loader

    @staticmethod
    def _reshape_inputs(inputs):
        n_devices = jax.local_device_count()
        if inputs.shape[0] % n_devices != 0:
            raise ValueError(
                f"The size of all batches of inputs must be a multiple of {n_devices}, that is the number of "
                f"available devices. However, a batch of inputs with shape {inputs.shape[0]} was found. "
                f"Please set an appropriate batch size."
            )
        return inputs.reshape((n_devices, -1) + inputs.shape[1:])

    def __iter__(self, *args, **kwargs):
        inputs_loader = map(
            lambda inputs: self._reshape_inputs(inputs), self._inputs_loader
        )
        inputs_loader = jax_utils.prefetch_to_device(inputs_loader, 2)
        yield from inputs_loader
