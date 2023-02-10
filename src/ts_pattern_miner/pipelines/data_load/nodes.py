import datetime as dt
import logging
import operator
import re
from functools import reduce
from typing import Dict
from typing import List, Callable

import pandas as pd
import xarray as xr
import dask
import dask.dataframe as dd
from ts_pattern_miner.exceptions import DataRelatedError
from ts_pattern_miner.utils import _stack_vectors_columns

logger = logging.getLogger("kedro")


logger = logging.getLogger("kedro")


def isin(series: pd.Series, value: List) -> pd.Series:
    return series.isin(value)


CUSTOM_OPERATORS = {
    "isin": isin,
}


def retype_str_to_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def retype_date_to_datetime(value: dt.date) -> dt.datetime:
    return dt.datetime(value.year, value.month, value.day)


CUSTOM_VTYPE_CONVERTERS = {"str2date": retype_str_to_date, "date2datetime": retype_date_to_datetime}

# ------------------
# Nodes


# def load_partitions_pandas(named_df: Dict[str, Callable], chunksize):
#     all_ddf_parts = []
#     for df_name, callable_df in named_df.items():
#         df_part = callable_df()
#         ddf_part = dd.from_pandas(df_part, chunksize=chunksize)
#         all_ddf_parts.append(ddf_part)
#     df = dd.concat(all_ddf_parts)
#     df = df.repartition(int(len(df) / chunksize))
#     return df


def load_partitions_pandas(named_df: Dict[str, Callable]):
    all_df_parts = []
    for df_name, callable_df in named_df.items():
        df_part = callable_df()
        all_df_parts.append(df_part)
    df = pd.concat(all_df_parts)
    return df

def convert_timestamp_to_datetime(
    df: pd.DataFrame,
    timestamp_column,
) -> pd.DataFrame:

    df[timestamp_column] = df[timestamp_column].map(lambda x: dt.datetime.fromtimestamp(x / 1e3))

    # if desired_frequency:
    #     df[timestamp_column] = df[timestamp_column].asfreq(desired_frequency)

    return df


def filter_rows_on_configured_rules(df: pd.DataFrame, split_rules: List[Dict]) -> pd.DataFrame:
    """
    For each rule defined in conf/base/parameters/dataprep , apply the rule to filter rows.
    :param df: extract dataframe from bigquery, output of data_download pipeline.
    :param split_rules:
    :return:
    """
    condition = pd.Series([True] * len(df), index=df.index)

    for filter_rule in split_rules:
        column_name = filter_rule["col"]
        if filter_rule.get("operator"):
            try:

                operation = getattr(operator, filter_rule["operator"])
            except AttributeError:
                operation = CUSTOM_OPERATORS[filter_rule["operator"]]

            value = filter_rule.get("value")
            conversion = filter_rule.get("conversion")
            if value is not None:
                if conversion is not None:
                    value = CUSTOM_VTYPE_CONVERTERS[conversion](value)

                condition = condition & operation(df[column_name], value)

                if filter_rule.get("keep_na", False):
                    condition = condition | df[column_name].isna()
                else:
                    condition = condition & ~df[column_name].isna()

    if df[condition].empty:
        raise DataRelatedError(f"Conditions led to an empty dataframe: {split_rules} ")
    else:
        df = df[condition]
        logger.debug(f"Creative score input data shape: {str(df.shape)} ")
    df = df.fillna(0)
    return df


def set_df_index(df: pd.DataFrame, index_cols: List[str]):
    # assert not df[index_cols].duplicated().sum(), "Duplicates found in index"
    df = df.set_index(index_cols)
    return df


def select_columns_from_regex(df: pd.DataFrame, regex_colnames: List[str]):
    """

    :param df: full dataframe
    :param cols_prefix_list: columns matching prefixes will ne kept
    :return: dataframe of creative features
    """

    all_cols = []
    for regex in regex_colnames:
        pattern = re.compile(regex)
        cols = [c for c in df.columns if re.match(pattern, c)]
        all_cols += cols
    all_cols = list(set(all_cols))
    return df[all_cols]


#
#

#
#
#
# def groupby_expected_frequency(df: pd.DataFrame, date_column_name: str, freq: str) -> pd.DataFrame:
#     df[date_column_name] = df[date_column_name].map(lambda s: s.dt.round(freq))
#     df = df.set_index(date_column_name)
#     df = df.groupby(date_column_name).apply(lambda grp: grp.mean(numeric_only=True))
#     return df


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

    # def convert_dataframe_to_xarray(df):
    #     xr.Dataset.from_dataframe(df)
    #     return xr.Dataset(df)

    array_symbols = df_symbols.to_xarray().chunk(chunks=dask_chunk_size)
    symbols_metrics = list(array_symbols.data_vars.keys())
    symbols, metrics = zip(*symbols_metrics)
    unique_symbols, unique_metrics = map(lambda x: sorted(list(set(x))), [symbols, metrics])

    # for each symbol, map the corresponding data_variables keys
    symbol_to_variables = {
        symbol: sorted(
            [variable for variable in symbols_metrics if variable[0] == symbol], key=lambda x: x[1]
        )
        for symbol in unique_symbols
    }

    assert all(
        len(l) == len(unique_metrics) for l in symbol_to_variables.values()
    ), "All symbols do not have the same number of metrics."

    coords = dict(metric=unique_metrics, timestamp=array_symbols.timestamp.data)

    # stack all metrics vectors (open, high,..) in a single array and build a new data_vars from it
    data_vars = {}
    for symbol in unique_symbols:
        symbol_metrics = array_symbols[symbol_to_variables[symbol]].data_vars.values()
        stacked_vectors = _stack_vectors_columns(
            [dv.data for dv in symbol_metrics], dask_chunk_size
        )
        stacked_vectors = stacked_vectors.persist()

        data_vars.update({symbol: (["timestamp", "metric"], stacked_vectors)})

    symbols_xarray = xr.Dataset(data_vars=data_vars, coords=coords)

    return symbols_xarray
