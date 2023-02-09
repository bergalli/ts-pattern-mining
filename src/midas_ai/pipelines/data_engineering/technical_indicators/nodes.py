from typing import List

import xarray as xr

from midas_ai.utils import _stack_vectors_columns
from midas_ai.pipelines.data_engineering.technical_indicators.dask_talib.ti_generators import generate_atr_dask_arrays, generate_sma_dask_arrays

TI_GEN_FUNS = {"sma": generate_sma_dask_arrays, "atr": generate_atr_dask_arrays}


def get_symbols_technical_indicators(
    symbols_xarray: xr.Dataset,
    ti_conf,
    dask_chunk_size,
) -> List[xr.Dataset]:
    all_symbols_indicators = []
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
    return all_symbols_indicators


def combine_features_into_dataframe(
    symbols_xarray: xr.Dataset,
    all_symbols_ti_arrays: List[xr.Dataset],
) -> xr.Dataset:
    features_xr = xr.merge([symbols_xarray] + all_symbols_ti_arrays)

    assert all(
        len(idx) == len(set(idx)) for idx in features_xr.indexes.values()
    ), "Some coordinates are duplicated in the final features_df"  # , cannot pivot."

    # noinspection PyTypeChecker
    features_xr = features_xr.persist()

    return features_xr
