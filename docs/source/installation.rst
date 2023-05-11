Installation
============
**NOTE:** Before installing Fortuna, you are required to `install JAX <https://github.com/google/jax#installation>`_ in your virtual environment.

You can install Fortuna by typing

.. code-block::

    pip install aws-fortuna

Alternatively, you can build the package using `Poetry <https://python-poetry.org/docs/>`_.
If you choose to pursue this way, first install Poetry and add it to your PATH
(see `here <https://python-poetry.org/docs/#installation>`_). Then type

.. code-block::

    poetry install

All the dependencies will be installed at their required versions.
If you also want to install the optional Sphinx dependencies to build the documentation,
add the flag :code:`-E docs` to the command above.
Finally, you can either access the virtualenv that Poetry created by typing :code:`poetry shell`,
or execute commands within the virtualenv using the :code:`run` command, e.g. :code:`poetry run python`.
