from typing import List

import numpy as np
import pandas as pd


# TODO optimize by chunking features_df and parrallel run this function
#  then count the most common dtypes for each columns or the max or smthg like that
def auto_astype(df: pd.DataFrame, keep_as_is: List[str] = None) -> pd.DataFrame:
    """
    Correct the bigquery dtypes to the correct primitive python dtypes,
     and optimize dtype memory use
    :param df:
    :param keep_as_is:
    :return:
    """
    if keep_as_is is None:
        keep_as_is = []

    # if any, convert nan values to python None, so eval works
    df = df.fillna(np.nan).replace([np.nan], [None])
    # convert columns of dtype object, from bool or int as string and mixed with None, to the
    for c in df.columns[df.dtypes == 'O'].drop(keep_as_is, errors='ignore'):
        try:
            str_to_num = df[c].astype(str).map(eval)
        except:
            continue
        df[c] = str_to_num

    # convert dtypes to optimal pandas dtypes
    cols_subset = df.columns.drop(keep_as_is, errors='ignore')
    df[cols_subset] = df[cols_subset].convert_dtypes()

    # optimize the memory size of numeral columns
    is_numeral_dtype = lambda x: pd.api.types.is_float_dtype(x) | pd.api.types.is_integer_dtype(x)
    numeral_columns = df.columns[df.apply(is_numeral_dtype, axis=0)]
    numeral_columns = numeral_columns.drop(keep_as_is, errors='ignore')
    optimized_dtypes = optimize_df(df[numeral_columns])
    df = df.astype(optimized_dtypes)

    return df


def _mem_usage(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:  # we assume if not a features_df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
    return "Total memory usage: {:03.2f} MB".format(usage_mb)


def optimize_df(df):
    """ from https://github.com/gbletsch/optimizedf/blob/master/optimizedf/optimize_df.py """
    '''Optimize features_df downcasting int and float columns and
    turning object columns in categorical when they have less then 50%
    unique values of the total.
    (Don't deal with datetime columns.)
    Ripped from 'https://www.dataquest.io/blog/pandas-big-data'
    to deal with large pandas features_df without using parallel
    os distributed computing.
    Parameters
    ----------
    features_df: pandas.features_df
        features_df to be optimized
    Return
    ------
    dict with optimized column dtypes to use when reading from the database.
    '''

    print('Optimizing features_df...\n')

    # downcast int dtypes
    gl_int = df.select_dtypes(include=['int'])
    converted_int = gl_int.apply(pd.to_numeric, downcast='unsigned')

    # downcast float dtypes
    gl_float = df.select_dtypes(include=['float'])
    converted_float = gl_float.apply(pd.to_numeric, downcast='float')

    # deal with string columns
    gl_obj = df.select_dtypes(include=['object']).copy()
    converted_obj = pd.DataFrame()

    for col in gl_obj.columns:
        num_unique_values = len(gl_obj[col].unique())
        num_total_values = len(gl_obj[col])
        if num_unique_values / num_total_values < 0.5:
            converted_obj.loc[:, col] = gl_obj[col].astype('category')
        else:
            converted_obj.loc[:, col] = gl_obj[col]

    # join converted columns
    optimized_gl = df.copy()
    optimized_gl[converted_int.columns] = converted_int
    optimized_gl[converted_float.columns] = converted_float
    optimized_gl[converted_obj.columns] = converted_obj

    # make dict with optimized dtypes
    dtypes = optimized_gl.dtypes
    dtypes_col = dtypes.index
    dtypes_type = [i.name for i in dtypes.values]
    column_types = dict(zip(dtypes_col, dtypes_type))

    print('Original features_df size:', _mem_usage(df))
    print('Optimized features_df size:', _mem_usage(optimized_gl))

    return column_types
