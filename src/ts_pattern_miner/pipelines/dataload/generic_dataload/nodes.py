import datetime as dt
import operator
import logging
from functools import reduce
from typing import Dict, List, Callable
import re

import pandas as pd

from ts_pattern_miner.exceptions import DataRelatedError


def isin(series: pd.Series, value: List) -> pd.Series:
    return series.isin(value)


CUSTOM_OPERATORS = {
    "isin": isin,
}


def retype_str_to_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


CUSTOM_VTYPE_CONVERTERS = {"date": retype_str_to_date}


logger = logging.getLogger("kedro")


def load_dask_df_to_xarray(df):
    return df


def filter_rows_on_configured_rules(df: pd.DataFrame, split_rules: Dict[str, dict]) -> pd.DataFrame:
    """
    For each rule defined in conf/base/parameters/dataprep , apply the rule to filter rows.
    :param df: extract dataframe from bigquery, output of data_download pipeline.
    :param split_rules:
    :return:
    """
    condition = pd.Series([True] * len(df), index=df.index)

    for column_name, filter_rule in split_rules.items():
        if filter_rule.get("operator"):
            try:
                operation = getattr(operator, filter_rule["operator"])
            except AttributeError:
                operation = CUSTOM_OPERATORS[filter_rule["operator"]]

            value = filter_rule.get("value")
            value_type = filter_rule.get("value_type")
            if value is not None:
                if value_type is not None:
                    value = CUSTOM_VTYPE_CONVERTERS[value_type](value)

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
    assert not df[index_cols].duplicated().sum(), "Duplicates found in index"
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
# def set_date_dtypes(
#         df: pd.DataFrame,
#         date_columns_config
# ) -> pd.DataFrame:
#     for colname, date_config in date_columns_config.items():
#         raw_format: str = date_config["raw_format"]
#         if raw_format == "timestamp":
#             df[colname] = df[colname].map(lambda x: dt.datetime.fromtimestamp(x))
#         else:
#             # raw_format = '%Y-%m-%d' e.g.
#             df[colname] = df[colname].map(lambda x: dt.datetime.strptime(x, raw_format))
#
#         freq: str = date_config['freq']
#         df[colname] = df[colname].asfreq(freq)
#
#     return df
#
#
#
# def groupby_expected_frequency(df: pd.DataFrame, date_column_name: str, freq: str) -> pd.DataFrame:
#     df[date_column_name] = df[date_column_name].map(lambda s: s.dt.round(freq))
#     df = df.set_index(date_column_name)
#     df = df.groupby(date_column_name).apply(lambda grp: grp.mean(numeric_only=True))
#     return df

