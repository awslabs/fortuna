from copy import deepcopy
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
)

from flax.core import FrozenDict

from fortuna.typing import AnyKey


def nested_get(
    d: Union[FrozenDict[AnyKey, Any], Dict[AnyKey, Any]], keys: List[AnyKey]
) -> Dict[AnyKey, Any]:
    """
    Get a sub-nested dictionary from a dictionary `d`. `keys` is the sequence of keys leading to the nested
    sub-dictionary.

    :param d: Dict[AnyKey, Any]
        Nested dictionary.
    :param keys: List[AnyKey
        Sequence of keys.

    :return: Dict[AnyKey, Any]
        The sub-nested dictionary.
    """
    if len(keys) == 0:
        raise ValueError("`keys` should not be empty.")
    if len(keys) == 1:
        return d[keys[0]]
    return nested_get(d[keys[0]], keys[1:])


def nested_set(
    d: Dict[AnyKey, Any],
    key_paths: Tuple[List[AnyKey], ...],
    objs: Tuple[Any],
    allow_nonexistent: bool = False,
) -> Dict[AnyKey, Any]:
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
    :param allow_nonexisten: bool
        Whether to create sequence of keys that are not found in the input dictionary or throw an exception.
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
            elif allow_nonexistent:
                d2[key] = {}
                d2 = d2[key]
            else:
                raise KeyError(error_msg)
        if keys[-1] in d2 or allow_nonexistent:
            d2[keys[-1]] = o
        else:
            raise KeyError(error_msg)
    return d0


def nested_pair(
    d: Dict[AnyKey, Any],
    key_paths: Tuple[List[AnyKey]],
    objs: Tuple[Any],
    labels: Tuple[str, str],
) -> Dict[AnyKey, Any]:
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
    d: Dict[AnyKey, Any], key_paths: Tuple[List[AnyKey], ...], labels: Tuple[str, str]
) -> Tuple[Dict[AnyKey, Any], Dict[AnyKey, Any]]:
    """
    Form two dictionaries out of the initial dictionary `d`. In correspondence to the sequences of keys in `key_paths`,
     the first and second dictionary will get the values marked by `labels`. Otherwise, the first dictionary takes the
     values of `d`, while the second dictionary does not take any.

    :param d: Dict[AnyKey, Any]
        A nested dictionary.
    :param key_paths: Tuple[List[AnyKey]]
        Each element of the tuple is a sequence of keys defining the path to an item of the nested dictionary. In
        correspondence to each key path, the value must be a dictionary with `labels` as keys.
    :param labels: Tuple[str, str]
        Labels indicating the keys of the objects going into the first and second dictionaries, respectively.

    :return Tuple[Dict[AnyKey, Any], Dict[AnyKey, Any]]
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


def find_one_path_to_key(
    d: Dict[AnyKey, Any], key: AnyKey, keys: Tuple[AnyKey] = ()
) -> Tuple[AnyKey]:
    if hasattr(d, "items"):
        for k, v in d.items():
            if k == key:
                keys += (k,)
                return keys
            if isinstance(v, (FrozenDict, dict)):
                tmp = find_one_path_to_key(v, key, keys + (k,))
                if key in tmp:
                    return tmp
    return keys


def nested_update(
    d: Dict[AnyKey, Any], *updating_d: Dict[AnyKey, Any]
) -> Dict[AnyKey, Any]:
    """
    Update a nested dictionary `d` with the content of an other dictionary `updating_d`.
    """
    # from https://github.com/pydantic/pydantic/blob/9d631a3429a66f30742c1a52c94ac18ec6ba848d/pydantic/utils.py#L198
    updated_mapping = d.copy()
    for updating_mapping in updating_d:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = nested_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping
