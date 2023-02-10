from typing import Tuple, List, Dict

import dask.array as da
import xarray as xr

from .distribute_talib import simple_moving_average_dask, average_true_range_dask


def generate_sma_dask_arrays(metrics_xarray: xr.Dataset, sma_params) -> Tuple[List[da.Array], dict]:
    timeperiods = sma_params["timeperiods"]
    metrics = sma_params["metrics"]

    coords = dict(timestamp=metrics_xarray.timestamp.data, metric=[])
    sma_dask_arrays = []
    for metric in metrics:
        metric_xarray = metrics_xarray.loc[dict(metric=metric)]

        for timeperiod in timeperiods:
            sma = simple_moving_average_dask(
                vector_metric=metric_xarray.data, timeperiod=timeperiod
            )

            coords["metric"] = coords["metric"] + [f"SMA_{metric}_{str(timeperiod)}"]
            sma_dask_arrays = sma_dask_arrays + [sma]

    return sma_dask_arrays, coords


def generate_atr_dask_arrays(metrics_xarray: xr.Dataset, atr_params) -> Tuple[List[da.Array], dict]:
    timeperiods = atr_params["timeperiods"]

    hlc_xarray = metrics_xarray.loc[dict(metric=["high", "low", "close"])]
    coords = dict(timestamp=metrics_xarray.timestamp.data, metric=[])

    atr_dask_arrays = []
    for timeperiod in timeperiods:
        atr = average_true_range_dask(hlc_xarray.data, timeperiod)

        coords["metric"] = coords["metric"] + [f"ATR_{str(timeperiod)}"]
        atr_dask_arrays = atr_dask_arrays + [atr]

    return atr_dask_arrays, coords
