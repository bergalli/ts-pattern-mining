import dask.distributed
from dask import dataframe as dd


def repartition(df: dd.DataFrame,
                chunk_size: str) -> dd.DataFrame:
    df = df.repartition(partition_size=chunk_size)
    # todo replace by features_df.compute_current_divisions when live
    #  see https://github.com/dask/dask/pull/8517
    df = df.reset_index().set_index(df.index.name or 'index', sorted=True)
    return df


def persist(df: dd.DataFrame, dask_client: dask.distributed.Client) -> dd.DataFrame:
    df = dask_client.persist(df)
    return df


def repartition_persist(df: dd.DataFrame,
                        chunk_size: str,
                        dask_client: dask.distributed.Client) -> dd.DataFrame:
    df = repartition(df, chunk_size)
    df = persist(df, dask_client)
    return df
