import math
from functools import wraps

from pandas import DataFrame, concat


# Just a bunch of tools to prevent pandas from becoming
# a memory eating monster
class DataFrameWrapper:

    def __init__(self, df: DataFrame, chunk_size=1000):
        self.df = df
        self.chunk_size = chunk_size

    def apply(self, func, *args, **kwargs):
        result = []
        chunks = self.compute_chunks()
        for chunk in chunks:
            for _, row in chunk.itertuples():
                result.append(func(row))
        return result

    def __getattr__(self, method: str, *args, **kwargs):
        return self.chunking_decorator(
            getattr(DataFrame, method))

    def compute_chunks(self):
        return [self.df[i * self.chunk_size: (i + 1) * self.chunk_size]
                for i in range(0, math.ceil(
                len(self.df) / self.chunk_size) + 1)]

    def chunking_decorator(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            chunks = self.compute_chunks()
            return concat([
                func(chunk, *args, **kwargs)
                for chunk in chunks
            ])

        return wrapper


def chunked_df(df: DataFrame):
    """
    ex: features_df = chunked_df(features_df).merge(features_df, on='analysis_uuid', how='inner').reset_index(drop=True)
    :param df:
    :return:
    """
    return DataFrameWrapper(df)
