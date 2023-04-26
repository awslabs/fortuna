Data loader
===========
This section describes Fortuna's data loader functionalities. A :class:`~fortuna.data.loader.DataLoader` object
is an iterable of two-dimensional tuples of arrays (either `NumPy <https://numpy.org/doc/stable/>`__-arrays or `JAX-NumPy <https://jax.readthedocs.io/en/latest/jax.numpy.html>`__-arrays),
where the first components are input variables and the second components are target variables. If your dispose of a data loader
of `TensorFlow <https://www.tensorflow.org/guide/data>`__ or `PyTorch <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`__ tensors, or others, you can convert them into something digestible by Fortuna using
the appropriate :class:`~fortuna.data.loader.DataLoader` functionality
(check :meth:`~fortuna.data.loader.DataLoader.from_tensorflow_data_loader`, :meth:`~fortuna.data.loader.DataLoader.from_torch_data_loader`).

The data :class:`~fortuna.data.loader.DataLoader` also allows you to generate an :class:`~fortuna.data.loader.InputsLoader` or a
:class:`~fortuna.data.loader.TargetsLoader`, i.e. data loaders of only inputs and only targets variables, respectively
(check :meth:`~fortuna.data.loader.DataLoader.to_inputs_loader` and :meth:`~fortuna.data.loader.DataLoader.to_targets_loader`).
Additionally, you can convert a data loader into an array of inputs, an array of targets, or a tuple of input and target
arrays (check :meth:`~fortuna.data.loader.DataLoader.to_array_inputs`, :meth:`~fortuna.data.loader.DataLoader.to_array_targets` and :meth:`~fortuna.data.loader.DataLoader.to_array_data`).

.. _data_loader:

.. autoclass:: fortuna.data.loader.DataLoader

.. _inputs_loader:

.. autoclass:: fortuna.data.loader.InputsLoader

.. _targets_loader:

.. autoclass:: fortuna.data.loader.TargetsLoader