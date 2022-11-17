from copy import deepcopy
from typing import Dict, List, Tuple, Union

from flax.core import FrozenDict


def nested_get(d: Union[FrozenDict, Dict], keys: List) -> Dict:
    """
    Get a sub-nested dictionary from a dictionary `d`. `keys` is the sequence of keys leading to the nested
    sub-dictionary.

    :param d: Dict
        Nested dictionary.
    :param keys: List
        Sequence of keys.

    :return: Dict
        The sub-nested dictionary.
    """
    if len(keys) == 0:
        raise ValueError("`keys` should not be empty.")
    if len(keys) == 1:
        return d[keys[0]]
    return nested_get(d[keys[0]], keys[1:])


def nested_set(d: Dict, key_paths: Tuple[List], objs: Tuple) -> Dict:
    """
    Set the values of a nested dictionary for the specified sequences of keys.

    :param d: Dict
        A nested dictionary.
    :param key_paths: Tuple[List]
        Each element of the tuple is a sequence of keys defining the path to an item of the nested dictionary. The
        length of the tuple must be the same as the length of `objs`, as to each key path the corresponding object in
        `objs` will be assigned.
    :param objs: Tuple
        Each element of the tuple is an object that will be assigned to the item specified by the corresponding
        sequence of keys.
    """
    if type(key_paths) != tuple:
        raise TypeError("`key_paths` must be a tuple.")
    if type(objs) != tuple:
        raise TypeError("`values` must be a tuple.")
    if len(key_paths) != len(objs):
        raise ValueError("`key_paths` and `values` must have the same length.")

    d0 = deepcopy(d)

    for keys, o in zip(key_paths, objs):
        error_msg = f"The sequence `keys={keys}` was not found in `d`."
        d2 = d0
        for key in keys[:-1]:
            if key in d2:
                d2 = d2[key]
            else:
                raise KeyError(error_msg)
        if keys[-1] in d2:
            d2[keys[-1]] = o
        else:
            raise KeyError(error_msg)
    return d0


def nested_pair(
    d: Dict, key_paths: Tuple[List], objs: Tuple, labels: Tuple[str, str]
) -> Dict:
    """
    Replace the values of a nested dictionary at the specified sequences of keys `keys` with dictionaries including both
    the values in the original dictionary and the new objects in `objs`. The keys of these dictionaries are given by
    `labels`.

    :param d: Dict
        A nested dictionary.
    :param key_paths: Tuple[List]
        Each element of the tuple is a sequence of keys defining the path to an item of the nested dictionary. The
        length of the tuple must be the same as the length of `objs`, as to each key path the corresponding object in
        `objs` will be assigned.
    :param objs: Tuple
        Each element of the tuple is an object that will be included in the item specified by the corresponding
        sequence of keys.
    :param labels: Tuple[str, str]
        Labels for the values of `d` and of `objs`, respectively.
    """
    if type(key_paths) != tuple:
        raise TypeError("`key_paths` must be a tuple.")
    if type(objs) != tuple:
        raise TypeError("`values` must be a tuple.")
    if len(key_paths) != len(objs):
        raise ValueError("`key_paths` and `values` must have the same length.")
    if len(labels) != 2:
        raise ValueError("The length of `labels` must be exactly 2.")

    d0 = deepcopy(d)

    for keys, o in zip(key_paths, objs):
        error_msg = f"The sequence `keys={keys}` was not found in `d`."
        d2 = d0
        for key in keys[:-1]:
            if key in d2:
                d2 = d2[key]
            else:
                raise KeyError(error_msg)
        if keys[-1] in d2:
            d2[keys[-1]] = {labels[0]: d2[keys[-1]], labels[1]: o}
        else:
            raise KeyError(error_msg)
    return d0


def nested_unpair(
    d: Dict, key_paths: Tuple[List], labels: Tuple[str, str]
) -> Tuple[Dict, Dict]:
    """
    Form two dictionaries out of the initial dictionary `d`. In correspondence to the sequences of keys in `key_paths`,
     the first and second dictionary will get the values marked by `labels`. Otherwise, the first dictionary takes the
     values of `d`, while the second dictionary does not take any.

    :param d: Dict
        A nested dictionary.
    :param key_paths: Tuple[List]
        Each element of the tuple is a sequence of keys defining the path to an item of the nested dictionary. In
        correspondence to each key path, the value must be a dictionary with `labels` as keys.
    :param labels: Tuple[str, str]
        Labels indicating the keys of the objects going into the first and second dictionaries, respectively.

    :return Tuple[Dict, dict]
        The two dictionaries.
    """
    if type(key_paths) != tuple:
        raise TypeError("`key_paths` must be a tuple.")
    if len(labels) != 2:
        raise ValueError("The length of `labels` must be exactly 2.")

    d01 = deepcopy(d)
    d02 = dict()

    for keys in key_paths:
        error_msg = f"The sequence `keys={keys}` was not found in `d`."
        d21 = d01
        d22 = d02
        for key in keys[:-1]:
            if key in d21:
                d21 = d21[key]
                if key not in d22:
                    d22[key] = dict()
                d22 = d22[key]
            else:
                raise KeyError(error_msg)
        if keys[-1] in d21:
            d22[keys[-1]] = d21[keys[-1]][labels[1]]
            d21[keys[-1]] = d21[keys[-1]][labels[0]]
        else:
            raise KeyError(error_msg)

    return d01, d02
