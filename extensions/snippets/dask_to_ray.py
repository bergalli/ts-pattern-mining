import dask
import ray
from dask import array as da, dataframe as dd
from ray.util.dask import ray_dask_get
import xarray as xr

def raydataset_from_xr_dataarray(xarr: xr.DataArray):
    ray_ds = raytensor_from_dask_array(xarr.data)


def raytensor_from_dask_array(arr: da.Array):
    # rechunk to maximize the number of rows for all columns (dask optimized)
    arr = arr.rechunk(chunks=(dask.config.get("array.chunk-size"), *arr.shape[1:]))
    gen_ndarray = (part.compute(scheduler=ray_dask_get) for part in arr.partitions)
    return ray.data.from_numpy(gen_ndarray)


def raydataset_from_dask_dataframe(ddf: dd.DataFrame):
    gen_dataframe = (part.compute(scheduler=ray_dask_get) for part in ddf.partitions)
    return ray.data.from_pandas(gen_dataframe)
