from __future__ import annotations

from typing import (
    Optional,
    Tuple,
)

import numpy as np

from fortuna.data.loader.base import (
    BaseDataLoaderABC,
    BaseInputsLoader,
    BaseTargetsLoader,
)
from fortuna.data.loader.utils import IterableData
from fortuna.typing import (
    Array,
    Batch,
)


class DataLoader(BaseDataLoaderABC):
    @property
    def num_unique_labels(self) -> Optional[int]:
        if self._num_unique_labels is None:
            self._num_unique_labels = len(np.unique(self.to_array_targets()))
        return self._num_unique_labels

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
            iterable=IterableData.from_batch_array_data(
                data, batch_size=batch_size, shuffle=shuffle, prefetch=prefetch
            )
        )

    def to_inputs_loader(self) -> InputsLoader:
        """
        Reduce a data loader to an inputs loader.

        Returns
        -------
        InputsLoader
            The inputs loader derived from the data loader.
        """
        return InputsLoader(IterableData.data_loader_to_inputs_iterable(self))

    def to_targets_loader(self) -> TargetsLoader:
        """
        Reduce a data loader to a targets loader.

        Returns
        -------
        TargetsLoader
            The targets loader derived from the data loader.
        """
        return TargetsLoader(IterableData.data_loader_to_targets_iterable(self))

    def to_array_data(self) -> Batch:
        """
        Reduce a data loader to a tuple of input and target arrays.

        Returns
        -------
        Batch
            Tuple of input and target arrays.
        """
        inputs, targets = [], []
        for batch_inputs, batch_targets in self:
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
        for batch_inputs, batch_targets in self:
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
        for batch_inputs, batch_targets in self:
            targets.append(batch_targets)
        return np.concatenate(targets, 0)

    def chop(self, divisor: int) -> DataLoader:
        """
        Chop the last part of each batch of the data loader, to make sure the number od data points per batch divides
        `divisor`.

        Parameters
        ----------
        divisor : int
            Number of data points that each batched must divide.

        Returns
        -------
        DataLoader
            A data loader with chopped batches.
        """

        def fun():
            for inputs, targets in self:
                reminder = targets.shape[0] % divisor
                if reminder == 0:
                    yield inputs, targets
                elif targets.shape[0] > divisor:
                    yield inputs[:-reminder], targets[:-reminder]

        return self.from_callable_iterable(fun)

    def split(self, n_data: int) -> Tuple[DataLoader, DataLoader]:
        """
        Split a data loader into two data loaders.

        Parameters
        ----------
        n_data: int
            Number of data point after which the data loader should be split. The first returned data loader will
            contain exactly `n_data` data points. The second one will contain the remaining ones.

        Returns
        -------
        Tuple[DataLoader, DataLoader]
            The two data loaders made out of the original one.
        """

        def data_loader1():
            count = 0
            for inputs, targets in self:
                if count == n_data:
                    break
                if count + inputs.shape[0] <= n_data:
                    count += inputs.shape[0]
                    yield inputs, targets
                else:
                    inputs, targets = (
                        inputs[: n_data - count],
                        targets[: n_data - count],
                    )
                    count = n_data
                    yield inputs, targets

        def data_loader2():
            count = 0
            for inputs, targets in self:
                if count > n_data:
                    yield inputs, targets
                elif (count <= n_data) and (count + inputs.shape[0] > n_data):
                    count2 = count
                    count += inputs.shape[0]
                    inputs, targets = (
                        inputs[n_data - count2 :],
                        targets[n_data - count2 :],
                    )
                    yield inputs, targets
                else:
                    count += inputs.shape[0]

        return self.from_callable_iterable(data_loader1), self.from_callable_iterable(
            data_loader2
        )

    def sample(self, seed: int, n_samples: int) -> DataLoader:
        """
        Sample from the data loader, with replacement.

        Parameters
        ----------
        seed: int
            Random seed.
        n_samples: int
            Number of samples.

        Returns
        -------
        DataLoader
            A data loader made of the sampled data points.
        """

        def fun():
            rng = np.random.default_rng(seed)
            count = 0

            while True:
                for inputs, targets in self:
                    if count == n_samples:
                        break
                    idx = rng.choice(2, inputs.shape[0]).astype("bool")
                    inputs, targets = inputs[idx], targets[idx]
                    if count + inputs.shape[0] > n_samples:
                        inputs, targets = (
                            inputs[: n_samples - count],
                            targets[: n_samples - count],
                        )
                    count += inputs.shape[0]
                    if inputs.shape[0] > 0:
                        yield inputs, targets

                if count == n_samples:
                    break

        return self.from_callable_iterable(fun)

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Get the shape of the inputs in the data loader."""

        def fun():
            for inputs, targets in self:
                input_shape = inputs.shape[1:]
                break
            return input_shape

        return fun()


class InputsLoader(BaseInputsLoader):
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
            iterable=IterableData.from_array_data(
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
        for batch_inputs in self:
            inputs.append(batch_inputs)
        return np.concatenate(inputs, 0)

    def chop(self, divisor: int) -> InputsLoader:
        """
        Chop the last part of each batch of the inputs loader, to make sure the number od data points per batch divides
        `divisor`.

        Parameters
        ----------
        divisor : int
            Number of data points that each batched must divide.

        Returns
        -------
        InputsLoader
            An inputs loader with chopped batches.
        """

        def fun():
            for inputs in self:
                reminder = inputs.shape[0] % divisor
                if reminder == 0:
                    yield inputs
                elif inputs.shape[0] > divisor:
                    yield inputs[:-reminder]

        return self.from_callable_iterable(fun)

    def sample(self, seed: int, n_samples: int) -> InputsLoader:
        """
        Sample from the inputs loader, with replacement.

        Parameters
        ----------
        seed: int
            Random seed.
        n_samples: int
            Number of samples.

        Returns
        -------
        InputsLoader
            An inputs loader made of the sampled inputs.
        """

        def fun():
            rng = np.random.default_rng(seed)
            count = 0

            while True:
                for inputs in self:
                    if count == n_samples:
                        break
                    idx = rng.choice(2, inputs.shape[0]).astype("bool")
                    inputs = inputs[idx]
                    if count + inputs.shape[0] > n_samples:
                        inputs = inputs[: n_samples - count]
                    count += inputs.shape[0]
                    if inputs.shape[0] > 0:
                        yield inputs

                if count == n_samples:
                    break

        return self.from_callable_iterable(fun)

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Get the shape of the inputs in the inputs loader."""

        def fun():
            for inputs in self:
                input_shape = inputs.shape[1:]
                break
            return input_shape

        return fun()

    def split(self, n_data: int) -> Tuple[InputsLoader, InputsLoader]:
        """
        Split an inputs loader into two inputs loaders.

        Parameters
        ----------
        n_data: int
            Number of data point after which the inputs loader should be split. The first returned inputs loader will
            contain exactly `n_data` inputs. The second one will contain the remaining ones.

        Returns
        -------
        Tuple[InputsLoader, InputsLoader]
            The two inputs loaders made out of the original one.
        """

        def inputs_loader1():
            count = 0
            for inputs in self:
                if count == n_data:
                    break
                if count + inputs.shape[0] <= n_data:
                    count += inputs.shape[0]
                    yield inputs
                else:
                    inputs = inputs[: n_data - count]
                    count = n_data
                    yield inputs

        def inputs_loader2():
            count = 0
            for inputs in self:
                if count > n_data:
                    yield inputs
                elif (count <= n_data) and (count + inputs.shape[0] > n_data):
                    count2 = count
                    count += inputs.shape[0]
                    inputs = inputs[n_data - count2 :]
                    yield inputs
                else:
                    count += inputs.shape[0]

        return self.from_callable_iterable(inputs_loader1), self.from_callable_iterable(
            inputs_loader2
        )


class TargetsLoader(BaseTargetsLoader):
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
            iterable=IterableData.from_array_data(
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
        for batch_targets in self:
            targets.append(batch_targets)
        return np.concatenate(targets, 0)

    def chop(self, divisor: int) -> TargetsLoader:
        """
        Chop the last part of each batch of the targets loader, to make sure the number od data points per batch divides
        `divisor`.

        Parameters
        ----------
        divisor : int
            Number of data points that each batched must divide.

        Returns
        -------
        TargetsLoader
            A targets loader with chopped batches.
        """

        def fun():
            for targets in self:
                reminder = targets.shape[0] % divisor
                if reminder == 0:
                    yield targets
                elif targets.shape[0] > divisor:
                    yield targets[:-reminder]

        return self.from_callable_iterable(fun)

    def sample(self, seed: int, n_samples: int) -> TargetsLoader:
        """
        Sample from the targets loader, with replacement.

        Parameters
        ----------
        seed: int
            Random seed.
        n_samples: int
            Number of samples.

        Returns
        -------
        TargetsLoader
            A targets loader made of the sampled targets.
        """

        def fun():
            rng = np.random.default_rng(seed)
            count = 0

            while True:
                for targets in self:
                    if count == n_samples:
                        break
                    idx = rng.choice(2, targets.shape[0]).astype("bool")
                    targets = targets[idx]
                    if count + targets.shape[0] > n_samples:
                        targets = targets[: n_samples - count]
                    count += targets.shape[0]
                    if targets.shape[0] > 0:
                        yield targets

                if count == n_samples:
                    break

        return self.from_callable_iterable(fun)

    def split(self, n_data: int) -> Tuple[TargetsLoader, TargetsLoader]:
        """
        Split a targets loader into two targets loaders.

        Parameters
        ----------
        n_data: int
            Number of data point after which the targets loader should be split. The first returned targets loader will
            contain exactly `n_data` targets. The second one will contain the remaining ones.

        Returns
        -------
        Tuple[TargetsLoader, TargetsLoader]
            The two targets loaders made out of the original one.
        """

        def targets_loader1():
            count = 0
            for targets in self:
                if count == n_data:
                    break
                if count + targets.shape[0] <= n_data:
                    count += targets.shape[0]
                    yield targets
                else:
                    targets = targets[: n_data - count]
                    count = n_data
                    yield targets

        def targets_loader2():
            count = 0
            for targets in self:
                if count > n_data:
                    yield targets
                elif (count <= n_data) and (count + targets.shape[0] > n_data):
                    count2 = count
                    count += targets.shape[0]
                    targets = targets[n_data - count2 :]
                    yield targets
                else:
                    count += targets.shape[0]

        return self.from_callable_iterable(
            targets_loader1
        ), self.from_callable_iterable(targets_loader2)
