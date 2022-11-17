from typing import Callable, Generator, Tuple

import numpy as np


def make_array_random_inputs(
    n_inputs: int, shape_inputs: Tuple[int, ...]
) -> np.ndarray:
    return np.random.normal(size=(n_inputs,) + shape_inputs)


def make_generator_fun_random_inputs(
    batch_size: int, n_batches: int, shape_inputs: Tuple[int, ...]
) -> Callable[[], Generator[np.ndarray, None, None]]:
    def inner():
        for i in range(n_batches):
            yield np.random.normal(size=(batch_size,) + shape_inputs)

    return inner


def make_generator_random_inputs(
    batch_size: int, n_batches: int, shape_inputs: Tuple[int, ...]
) -> Generator[np.ndarray, None, None]:
    for i in range(n_batches):
        yield np.random.normal(size=(batch_size,) + shape_inputs)


def make_array_random_targets(n_inputs: int, output_dim: int, output_type: str):
    if output_type not in ["discrete", "continuous"]:
        raise Exception(
            "`output_type={}` not recognized. Please choose among the following list: {}.".format(
                "discrete", "continuous"
            )
        )
    return (
        np.random.normal(size=(n_inputs, output_dim))
        if output_type == "continuous"
        else np.random.choice(output_dim, size=n_inputs)
    )


def make_generator_fun_random_targets(
    batch_size: int, n_batches: int, output_dim: int, output_type: str,
) -> Callable[[], Generator[Tuple[np.ndarray, np.ndarray], None, None]]:
    if output_type not in ["discrete", "continuous"]:
        raise Exception(
            "`output_type={}` not recognized. Please choose among the following list: {}.".format(
                "discrete", "continuous"
            )
        )

    def inner():
        for i in range(n_batches):
            yield np.random.normal(
                size=(batch_size, output_dim)
            ) if output_type == "continuous" else np.random.choice(
                output_dim, size=batch_size
            )

    return inner


def make_generator_random_targets(
    batch_size: int, n_batches: int, output_dim: int, output_type: str,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    if output_type not in ["discrete", "continuous"]:
        raise Exception(
            "`output_type={}` not recognized. Please choose among the following list: {}.".format(
                "discrete", "continuous"
            )
        )
    for i in range(n_batches):
        yield np.random.normal(
            size=(batch_size, output_dim)
        ) if output_type == "continuous" else np.random.choice(
            output_dim, size=batch_size
        )


def make_array_random_data(
    n_data: int, shape_inputs: Tuple[int, ...], output_dim: int, output_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    if output_type not in ["discrete", "continuous"]:
        raise Exception(
            "`output_type={}` not recognized. Please choose among the following list: {}.".format(
                "discrete", "continuous"
            )
        )
    return (
        np.random.normal(size=(n_data,) + shape_inputs),
        np.random.normal(size=(n_data, output_dim))
        if output_type == "continuous"
        else np.random.choice(output_dim, size=n_data),
    )


def make_generator_fun_random_data(
    batch_size: int,
    n_batches: int,
    shape_inputs: Tuple[int],
    output_dim: int,
    output_type: str,
) -> Callable[[], Generator[Tuple[np.ndarray, np.ndarray], None, None]]:
    if output_type not in ["discrete", "continuous"]:
        raise Exception(
            "`output_type={}` not recognized. Please choose among the following list: {}.".format(
                "discrete", "continuous"
            )
        )

    def inner():
        for i in range(n_batches):
            yield np.random.normal(size=(batch_size,) + shape_inputs), np.random.normal(
                size=(batch_size, output_dim)
            ) if output_type == "continuous" else np.random.choice(
                output_dim, size=batch_size
            )

    return inner


def make_generator_random_data(
    batch_size: int,
    n_batches: int,
    shape_inputs: Tuple[int],
    output_dim: int,
    output_type: str,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    if output_type not in ["discrete", "continuous"]:
        raise Exception(
            "`output_type={}` not recognized. Please choose among the following list: {}.".format(
                "discrete", "continuous"
            )
        )
    for i in range(n_batches):
        yield np.random.normal(size=(batch_size,) + shape_inputs), np.random.normal(
            size=(batch_size, output_dim)
        ) if output_type == "continuous" else np.random.choice(
            output_dim, size=batch_size
        )
