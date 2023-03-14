# Fortuna Documentation

The documentation for Fortuna can be found [here](https://aws-fortuna.readthedocs.io/en/latest/).

## Build the documentation

The documentation for Fortuna is comprised of a series of notebooks. To serve these locally, there are two steps that need to be taken that we outline below.

### Prerequisites

To build the documentation, first install Fortuna and it's dependencies by following the [installation instructions](https://github.com/awslabs/fortuna#installation). Next, install the documentation requirements through
```bash
poetry install -E docs
```
Finally, install `jupytext` through either pip or conda, details can be found [here](https://github.com/mwouts/jupytext#install). 

### Notebooks

For easier version control, the notebooks are stored as `.pct.py` files. To convert these to `.ipynb` files, run the following command from the root of the repository:

```bash
jupytext --to notebook examples/*pct.py
```

This will create a corresponding notebook file for each `.pct.py` file that can be opened in Jupyter.

### Additional Information

For [VSCode](https://code.visualstudio.com/) users, we recommend installing the [Jupytext extension](https://marketplace.visualstudio.com/items?itemName=congyiwu.vscode-jupytext) to automatically render `.pct.py` as Jupyter notebooks when opened in VSCode.