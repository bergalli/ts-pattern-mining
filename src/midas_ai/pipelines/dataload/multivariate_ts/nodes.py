import pandas as pd
import logging
from typing import List, Callable
from functools import reduce

import xarray as xr

from midas_ai.utils import _stack_vectors_columns

logger = logging.getLogger("kedro")


def pivot_foreach_label(
    df: pd.DataFrame,
    index_column: str,
    labels_column: str,
    values_columns: List[str],
) -> pd.DataFrame:
    dfs = []
    for label in df[labels_column].unique():
        logger.info(f"Splitting dataframe on label {label} of column {labels_column}")
        sub_df = df.loc[
            df[labels_column] == label, [index_column, labels_column] + list(values_columns)
        ]
        sub_df = sub_df.drop(labels_column, axis=1)

        sub_df = sub_df.sort_values(index_column)
        sub_df = sub_df.set_index(index_column)

        sub_df.columns = pd.MultiIndex.from_tuples(
            zip([label] * len(sub_df.columns), sub_df.columns), names=["symbol", "metric"]
        )
        dfs.append(sub_df)

    logger.info("Merging splitted dataframes")
    merge_fun: Callable = lambda df1, df2: pd.merge(
        df1, df2, how="outer", left_index=True, right_index=True
    )
    df = reduce(merge_fun, dfs)

    return df


def convert_ddf_to_xarray(df_symbols: pd.DataFrame, dask_chunk_size) -> xr.Dataset:
    """
    From a modin dataframe with MultiIndex columns, creates a xarray dataset with the symbols as
    data variables keys, and the metrics and timestamp as coordinates/dimensions
    Args:
        df_symbols:
        dask_chunk_size:
    Returns:

    """

    array_symbols = df_symbols.to_xarray().chunk(chunks=dask_chunk_size)
    symbols_metrics = list(array_symbols.data_vars.keys())
    symbols, metrics = zip(*symbols_metrics)
    unique_symbols, unique_metrics = map(lambda x: sorted(list(set(x))), [symbols, metrics])

    # for each symbol, map the corresponding data_variables keys
    symbol_to_variables = {
        symbol: sorted([variable for variable in symbols_metrics if variable[0] == symbol], key=lambda x: x[1])
        for symbol in unique_symbols
    }

    assert all(len(l) == len(unique_metrics) for l in symbol_to_variables.values()) \
        , "All symbols do not have the same number of metrics."

    coords = dict(
        metric=unique_metrics,
        timestamp=array_symbols.timestamp.data
    )

    # stack all metrics vectors (open, high,..) in a single array and build a new data_vars from it
    data_vars = {}
    for symbol in unique_symbols:
        symbol_metrics = array_symbols[symbol_to_variables[symbol]].data_vars.values()
        stacked_vectors = _stack_vectors_columns([dv.data for dv in symbol_metrics], dask_chunk_size)
        stacked_vectors = stacked_vectors.persist()

        data_vars.update({
            symbol: (["timestamp", "metric"], stacked_vectors)
        })

    symbols_xarray = xr.Dataset(data_vars=data_vars, coords=coords)

    return symbols_xarray