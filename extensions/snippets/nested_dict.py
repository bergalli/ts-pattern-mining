from functools import reduce
from typing import Iterable, Union, Any


def recursive_set_dict(
        d: dict,
        keys: Iterable[Union[str, int]],
        value: Any
):
    d_c = d
    for k in keys[:-1]:
        d_c = d_c.setdefault(k, {})
    d_c[keys[-1]] = value
    return None


def recursive_get_dict(
        d: dict,
        keys: Iterable[Union[str, int]]
):
    try:
        return reduce(lambda c, k: c[k], keys, d)
    except KeyError:
        raise Exception(f"{keys} not found")
