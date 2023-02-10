from typing import List, Tuple

import dask.array as da
import numpy as np
import ray
import xarray as xr

from ts_pattern_miner.utils import _stack_vectors_columns
from .dask_talib.ti_generators import (
    generate_atr_dask_arrays,
    generate_sma_dask_arrays,
)

TI_GEN_FUNS = {"sma": generate_sma_dask_arrays, "atr": generate_atr_dask_arrays}


def get_symbols_technical_indicators(
    symbols_xarray: xr.Dataset,
    ti_conf,
    dask_chunk_size,
) -> xr.Dataset:
    all_symbols_indicators = [symbols_xarray]
    for ti_name, ti_params in ti_conf.items():
        generate_ti_fun = TI_GEN_FUNS[ti_name]

        symbols_indicator = []
        for symbol, metrics_xarray in symbols_xarray.items():
            # sma_xarrays_as_dict = {'timestamp': {'dims': 'timestamp', 'data': metrics_xarray.timestamp.data}}

            ti_dask_arrays, coords = generate_ti_fun(metrics_xarray, ti_params)

            stacked_vectors = _stack_vectors_columns(ti_dask_arrays, dask_chunk_size)
            data_vars = {symbol: (["timestamp", "metric"], stacked_vectors)}
            symbol_ti_xarray = xr.Dataset(data_vars=data_vars, coords=coords)

            symbols_indicator.append(symbol_ti_xarray)
        all_symbols_indicators += symbols_indicator

    features_xr = xr.merge(all_symbols_indicators)
    features_xr = features_xr.persist()
    return features_xr


def shift_features(features_xr: xr.Dataset, shift_interval: Tuple[int, int]) -> xr.Dataset:
    def shift_array(xr_df: xr.Dataset, period: int, fill_value=np.NaN) -> xr.Dataset:
        """
        Convert roll the data
        """
        shifted_data_vars = {}
        for symbol, dv in xr_df.data_vars.items():
            rolled = da.roll(dv.data, period, axis=0)

            if period < 0:
                rolled[period:, ::] = fill_value
            elif period > 0:
                rolled[:period, ::] = fill_value

            shifted_data_vars.update({symbol: (["timestamp", "metric"], rolled)})
        shifted_array_xr = xr.Dataset(data_vars=shifted_data_vars, coords=xr_df.coords)

        return shifted_array_xr

    features_shifts_xr = {
        period: shift_array(features_xr, period)
        for period in range(shift_interval[0], shift_interval[1])
    }
    # create new dimension "shift"
    features_shifts_xr = [
        arr.expand_dims("shift").assign_coords(dict(shift=[shift_id]))
        for shift_id, arr in features_shifts_xr.items()
    ]
    features_shifts_xr = xr.concat(features_shifts_xr, dim="shift")

    features_shifts_xr = features_shifts_xr.transpose("timestamp", "shift", "metric")

    features_shifts_xr.persist()

    return features_shifts_xr


def select_targets_columns(symbols_xarray: xr.Dataset, target_metrics: List[str]) -> xr.Dataset:
    """
    Same operation as node combine_features_into_dataframe()
    Args:
        symbols_xarray:
        target_metrics:

    Returns:

    """
    targets_xr = symbols_xarray.loc[dict(metric=target_metrics)]
    # targets_xr = targets_xr.assign_coords(dict(role=["target"]))

    return targets_xr
