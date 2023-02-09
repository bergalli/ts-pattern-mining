
import pandas as pd
import xarray as xr
from typing import Union

def multi_index_merge(
        *objects: Union[xr.Dataset, xr.DataArray],
        **merge_kwargs):

    for k, v in a.indexes.items():
        if isinstance(v, pd.MultiIndex):
            flatten_dims = dict(zip(v.names, zip(*v.to_list())))
            dims = dict(zip(v.to_list(), range(len(v))))
    ads.reset_index(dims_or_levels="features")
    objects_index_reseted = [obj.reset_index() for obj in objects]
    return xr.merge(objects=objects_index_reseted, **merge_kwargs)
