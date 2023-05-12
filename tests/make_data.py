from typing import (
    Callable,
    Generator,
    Optional,
    Tuple,
)

import numpy as np


def _check_output_type(output_type):
    if output_type not in ["discrete", "continuous"]:
        raise Exception(
            f"`output_type={output_type}` not recognized. "
            "Please choose among the following list: ['discrete', 'continuous']"
        )


def make_array_random_inputs(
    n_inputs: int, shape_inputs: Tuple[int, ...], seed: Optional[int] = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_inputs,) + shape_inputs)


def make_generator_fun_random_inputs(
    batch_size: int,
    n_batches: int,
    shape_inputs: Tuple[int, ...],
    seed: Optional[int] = 0,
) -> Callable[[], Generator[np.ndarray, None, None]]:
    rng = np.random.default_rng(seed)

    def inner():
        for i in range(n_batches):
            yield rng.normal(size=(batch_size,) + shape_inputs)

    return inner


def make_generator_random_inputs(
    batch_size: int,
    n_batches: int,
    shape_inputs: Tuple[int, ...],
    seed: Optional[int] = 0,
) -> Generator[np.ndarray, None, None]:
    rng = np.random.default_rng(seed)
    for i in range(n_batches):
        yield rng.normal(size=(batch_size,) + shape_inputs)


def make_array_random_targets(
    n_inputs: int, output_dim: int, output_type: str, seed: Optional[int] = 0
) -> np.ndarray:
    _check_output_type(output_type)
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n_inputs, output_dim))
        if output_type == "continuous"
        else rng.choice(output_dim, size=n_inputs)
    )


def make_generator_fun_random_targets(
    batch_size: int,
    n_batches: int,
    output_dim: int,
    output_type: str,
    seed: Optional[int] = 0,
) -> Callable[[], Generator[Tuple[np.ndarray, np.ndarray], None, None]]:
    _check_output_type(output_type)
    rng = np.random.default_rng(seed)

    def inner():
        for i in range(n_batches):
            yield rng.normal(
                size=(batch_size, output_dim)
            ) if output_type == "continuous" else rng.choice(
                output_dim, size=batch_size
            )

    return inner


def make_generator_random_targets(
    batch_size: int,
    n_batches: int,
    output_dim: int,
    output_type: str,
    seed: Optional[int] = 0,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    _check_output_type(output_type)
    rng = np.random.default_rng(seed)
    for i in range(n_batches):
        yield rng.normal(
            size=(batch_size, output_dim)
        ) if output_type == "continuous" else rng.choice(output_dim, size=batch_size)


def make_array_random_data(
    n_data: int,
    shape_inputs: Tuple[int, ...],
    output_dim: int,
    output_type: str,
    seed: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    _check_output_type(output_type)
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n_data,) + shape_inputs),
        rng.normal(size=(n_data, output_dim))
        if output_type == "continuous"
        else rng.choice(output_dim, size=n_data),
    )


def make_generator_fun_random_data(
    batch_size: int,
    n_batches: int,
    shape_inputs: Tuple[int],
    output_dim: int,
    output_type: str,
    seed: Optional[int] = 0,
) -> Callable[[], Generator[Tuple[np.ndarray, np.ndarray], None, None]]:
    _check_output_type(output_type)
    rng = np.random.default_rng(seed)

    def inner():
        for i in range(n_batches):
            yield rng.normal(size=(batch_size,) + shape_inputs), rng.normal(
                size=(batch_size, output_dim)
            ) if output_type == "continuous" else rng.choice(
                output_dim, size=batch_size
            )

    return inner


def make_generator_random_data(
    batch_size: int,
    n_batches: int,
    shape_inputs: Tuple[int],
    output_dim: int,
    output_type: str,
    seed: Optional[int] = 0,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    _check_output_type(output_type)
    rng = np.random.default_rng(seed)
    for i in range(n_batches):
        yield rng.normal(size=(batch_size,) + shape_inputs), rng.normal(
            size=(batch_size, output_dim)
        ) if output_type == "continuous" else rng.choice(output_dim, size=batch_size)
