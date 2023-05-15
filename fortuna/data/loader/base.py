from __future__ import annotations

import abc
from typing import (
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from flax import jax_utils
import jax
from jax.tree_util import tree_map

from fortuna.data.loader.utils import IterableData
from fortuna.typing import (
    Array,
    Batch,
    InputData,
    Shape,
    Status,
    Targets,
)

T = TypeVar("T")


class BaseDataLoaderABC(abc.ABC):
    def __init__(self, iterable: IterableData, num_unique_labels: int = None):
        self._iterable = iterable
        self._num_unique_labels = num_unique_labels

    def __iter__(self) -> Iterable[Batch]:
        yield from self._iterable

    @property
    def size(self) -> int:
        """
        The number of data points in the data loader.

        Returns
        -------
        int
            Number of data points.
        """
        c = 0
        for inputs, targets in self:
            if isinstance(inputs, dict):
                inputs = inputs[list(inputs.keys())[0]]
            c += inputs.shape[0]
        return c

    @property
    @abc.abstractmethod
    def num_unique_labels(self) -> Optional[int]:
        """
        Number of unique target labels in the task (classification only)

        Returns
        -------
        int
            Number of unique target labels in the task if it is a classification one.
            Otherwise returns None.
        """
        return self._num_unique_labels

    @property
    @abc.abstractmethod
    def input_shape(self) -> Shape:
        """
        Get the shape of the inputs in the data loader.
        """
        pass

    @abc.abstractmethod
    def to_inputs_loader(self) -> BaseInputsLoader:
        """
        Reduce a data loader to an inputs loader.

        Returns
        -------
        BaseInputsLoader
            The inputs loader derived from the data loader. This will be a concrete instance of a subclass
            of :class:`~fortuna.data.loader.BaseInputsLoader`.
        """
        pass

    @abc.abstractmethod
    def to_targets_loader(self) -> BaseTargetsLoader:
        """
        Reduce a data loader to a targets loader.

        Returns
        -------
        BaseTargetsLoader
            The targets loader derived from the data loader. This will be a concrete instance of a subclass
            of :class:`~fortuna.data.loader.BaseTargetsLoader`.
        """
        pass

    @abc.abstractmethod
    def to_array_targets(self) -> Array:
        """
        Reduce a data loader to an array of target data.

        Returns
        -------
        Array
            Array of input data.
        """
        pass

    @classmethod
    def from_callable_iterable(
        cls: Type[T],
        fun: Callable[
            [],
            Iterable[Batch],
        ],
    ) -> T:
        """
        Transform a callable iterable into a concrete instance of a subclass of
        :class:`~fortuna.data.loader.BaseDataLoader`.

        Parameters
        ----------
        fun: Callable[[], Iterable[Batch]]
            A callable iterable of tuples of input and target arrays.

        Returns
        -------
        T
            A concrete instance of a subclass of :class:`~fortuna.data.loader.BaseDataLoader`.
        """
        return cls(iterable=IterableData.from_callable(fun))

    @classmethod
    def from_iterable(
        cls: Type[T],
        iterable: Iterable[Batch],
    ) -> T:
        """
        Transform an iterable into a concrete instance of a subclass of
        :class:`~fortuna.data.loader.BaseDataLoader`.

        Parameters
        ----------
        iterable: Iterable[Batch]
            An iterable of tuples of input and target arrays.

        Returns
        -------
        T
            A concrete instance of a subclass of :class:`~fortuna.data.loader.BaseDataLoader`.
        """
        return cls(iterable=IterableData.from_iterable(iterable))

    @classmethod
    def from_tensorflow_data_loader(cls: Type[T], tf_data_loader) -> T:
        """
        Transform a TensorFlow data loader into a concrete instance of a subclass of
        :class:`~fortuna.data.loader.BaseDataLoader`.

        Parameters
        ----------
        tf_data_loader
            A TensorFlow data loader where each batch is a tuple of input and target Tensors.

        Returns
        -------
        T
            A concrete instance of a subclass of :class:`~fortuna.data.loader.BaseDataLoader`.
        """
        return cls(iterable=IterableData.from_tf_dataloader(tf_data_loader))

    @classmethod
    def from_torch_data_loader(cls: Type[T], torch_data_loader) -> T:
        """
        Transform a PyTorch data loader into a concrete instance of a subclass of
        :class:`~fortuna.data.loader.BaseDataLoader`.

        Parameters
        ----------
        torch_data_loader
            A PyTorch data loader where each batch is a tuple of input and target Tensors.

        Returns
        -------
        T
            A concrete instance of a subclass of :class:`~fortuna.data.loader.BaseDataLoader`.
        """
        return cls(iterable=IterableData.from_torch_dataloader(torch_data_loader))

    @classmethod
    def from_inputs_loaders(
        cls: Type[T], inputs_loaders: List[BaseInputsLoader], targets: List[int]
    ) -> T:
        """
        Transform a list of inputs loader into a concrete instance of a subclass of
        :class:`~fortuna.data.loader.BaseDataLoader`. The newly created data loader is formed out
        of concatenated batches of inputs and the respective assigned target variable.

        Parameters
        ----------
        inputs_loaders: List[BaseInputsLoader]
            A list of inputs loaders.
        targets: List[int]
            A target variable for each inputs loader.

        Returns
        -------
        T
            A concrete instance of a subclass of :class:`~fortuna.data.loader.BaseDataLoader`.
            The data loader object is formed by the concatenated batches of inputs, and the assigned targets.
        """
        return cls(
            iterable=IterableData.inputs_loaders_to_batch_iterable(
                inputs_loaders=inputs_loaders, targets=targets
            )
        )

    def to_transformed_data_loader(
        self: T,
        transform: Callable[
            [InputData, Array, Status], Tuple[InputData, Array, Status]
        ],
        status: Optional[Status] = None,
    ) -> T:
        """
        Transform the batches of an existing data loader.

        Parameters
        ----------
        transform : Callable[[InputData, Array, Status], Tuple[InputData, Array, Status]]
            A transformation function. It takes a batch and returns its transformation. A status may be updated
            during the process.
        status : Optional[Status]
            An initial status. This may include pre-computed objects used by the transformation.

        Returns
        -------
        T
            A concrete instance of a subclass of :class:`~fortuna.data.loader.BaseDataLoader`.
        """
        cls = self.__class__
        return cls(
            IterableData.transform_data_loader(
                loader=self, transform_fn=transform, status=status
            )
        )


class BaseInputsLoader:
    def __init__(self, iterable):
        self._iterable = iterable

    def __iter__(self) -> Iterable[InputData]:
        yield from self._iterable

    @property
    def size(self) -> int:
        """
        The number of data points in the inputs loader.

        Returns
        -------
        int
            Number of data points.
        """
        c = 0
        for inputs in self:
            if isinstance(inputs, dict):
                inputs = inputs[list(inputs.keys())[0]]
            c += inputs.shape[0]
        return c

    @property
    def input_shape(self) -> Shape:
        """Get the shape of the inputs in the inputs loader."""

        def fun():
            for inputs in self:
                if isinstance(inputs, dict):
                    input_shape = {k: v.shape[1:] for k, v in inputs.items()}
                else:
                    input_shape = inputs.shape[1:]
                break
            return input_shape

        return fun()

    @classmethod
    def from_data_loader(cls: Type[T], data_loader: BaseDataLoaderABC) -> T:
        """
        Reduce a data loader to an inputs loader.

        Parameters
        ----------
        data_loader : DataLoader
            A data loader.

        Returns
        -------
        T
            A concrete instance of a subclass of :class:`~fortuna.data.loader.BaseInputsLoader`.
        """
        return cls(iterable=IterableData.data_loader_to_inputs_iterable(data_loader))

    @classmethod
    def from_callable_iterable(
        cls: Type[T],
        fun: Callable[[], Iterable[InputData]],
    ) -> T:
        """
        Transform a callable iterable into a concrete instance of a subclass of
        :class:`~fortuna.data.loader.BaseInputsLoader`

        Parameters
        ----------
        fun: Callable[[], Iterable[InputData]]
            A callable iterable of input data.

        Returns
        -------
        T
            A concrete instance of a subclass of :class:`~fortuna.data.loader.BaseInputsLoader`.
        """
        return cls(iterable=IterableData.from_callable(fun))

    @classmethod
    def from_iterable(
        cls: Type[T],
        iterable: Iterable[InputData],
    ) -> T:
        """
        Transform an iterable into a concrete instance of a subclass of
        :class:`~fortuna.data.loader.BaseInputsLoader`

        Parameters
        ----------
        iterable: Iterable[InputData]
            An iterable of input data.

        Returns
        -------
        T
            A concrete instance of a subclass of :class:`~fortuna.data.loader.BaseInputsLoader`.
        """
        return cls(iterable=IterableData.from_iterable(iterable))

    def to_transformed_inputs_loader(
        self: T,
        transform: Callable[[InputData, Status], Tuple[InputData, Status]],
        status: Optional[Status] = None,
    ) -> T:
        """
        From an existing loader of inputs, create a loader with transformed inputs.

        Parameters
        ----------
        transform : Callable[[Array, Status], Tuple[Array, Status]]
            A transformation function. It takes a batch of inputs and returns their transformation.
        status : Optional[Status]
            An initial status. This may include pre-computed objects used by the transformation.

        Returns
        -------
        T
            A concrete instance of a subclass of :class:`~fortuna.data.loader.BaseInputsLoader`.
        """
        cls = self.__class__
        return cls(
            IterableData.transform_inputs_or_targets_loader(
                loader=self, transform_fn=transform, status=status
            )
        )


class BaseTargetsLoader:
    """
    A targets loader class.
    """

    def __init__(self, iterable):
        self._iterable = iterable

    def __iter__(self) -> Iterable[Targets]:
        yield from self._iterable

    @property
    def size(self) -> int:
        """
        The number of data points in the targets loader.

        Returns
        -------
        int
            Number of data points.
        """
        c = 0
        for targets in self:
            c += targets.shape[0]
        return c

    @classmethod
    def from_data_loader(cls: Type[T], data_loader: BaseDataLoaderABC) -> T:
        """
        Reduce a data loader to a targets loader.

        Parameters
        ----------
        data_loader : DataLoader
            A data loader.

        Returns
        -------
        T
            A concrete instance of a subclass of :class:`~fortuna.data.loader.BaseTargetsLoader`.
        """
        return IterableData.data_loader_to_targets_iterable(data_loader)

    @classmethod
    def from_callable_iterable(
        cls: Type[T],
        fun: Callable[[], Iterable[Array]],
    ) -> T:
        """
        Transform a callable iterable into a concrete instance of a subclass of
        :class:`~fortuna.data.loader.BaseTargetsLoader`.

        Parameters
        ----------
        fun: Callable[[], Iterable[Union[Batch, InputData, Array]]],
            A callable iterable of target arrays.

        Returns
        -------
        T
            A concrete instance of a subclass of :class:`~fortuna.data.loader.BaseTargetsLoader`.
        """
        return cls(iterable=IterableData.from_callable(fun))

    @classmethod
    def from_iterable(
        cls: Type[T],
        iterable: Iterable[Array],
    ) -> T:
        """
        Transform an iterable into a concrete instance of a subclass of
        :class:`~fortuna.data.loader.BaseTargetsLoader`.

        Parameters
        ----------
        iterable: Iterable[Array]
            An iterable of target arrays.

        Returns
        -------
        T
           A concrete instance of a subclass of :class:`~fortuna.data.loader.BaseTargetsLoader`.
        """
        return cls(iterable=IterableData.from_iterable(iterable))

    def to_transformed_targets_loader(
        self: T,
        transform: Callable[[Array, Status], Tuple[Array, Status]],
        status: Optional[Status] = None,
    ) -> T:
        """
        From an existing loader of targets, create a loader with transformed targets.

        Parameters
        ----------
        transform : Callable[[Array, Status], Tuple[Array, Status]]
            A transformation function. It takes a batch of targets and returns their transformation. A status may be
            updated during the process.
        status : Optional[Status]
            An initial status. This may include pre-computed objects used by the transformation.

        Returns
        -------
        T
            A concrete instance of a subclass of :class:`~fortuna.data.loader.BaseTargetsLoader`.
        """
        cls = self.__class__
        return cls(
            IterableData.transform_inputs_or_targets_loader(
                loader=self, transform_fn=transform, status=status
            )
        )


class DeviceDimensionAugmentedLoader:
    def __init__(self, loader):
        self._loader = loader

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
        loader = map(lambda batch: tree_map(self._reshape_inputs, batch), self._loader)
        loader = jax_utils.prefetch_to_device(loader, 2)
        yield from loader
