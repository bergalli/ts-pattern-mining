from kedro.pipeline import Pipeline, pipeline, node


from .nodes import (
    load_partitions_pandas,
    convert_timestamp_to_datetime,
    filter_rows_on_configured_rules,
    set_df_index,
    select_columns_from_regex,
    pivot_foreach_label,
    convert_ddf_to_xarray,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                load_partitions_pandas,
                inputs=dict(named_df="daily_crypto_currencies"),
                outputs="df_raw",
            ),
            node(
                convert_timestamp_to_datetime,
                inputs=dict(df="df_raw", timestamp_column="params:timestamp_column"),
                outputs="df_date_typed",
            ),
            node(
                filter_rows_on_configured_rules,
                inputs=dict(df="df_date_typed", split_rules="params:data_split_rules"),
                outputs="df_filtered",
            ),
            node(
                select_columns_from_regex,
                inputs=dict(df="df_filtered", regex_colnames="params:regex_colnames"),
                outputs="df_columns_selected",
            ),
            node(
                set_df_index,
                inputs=dict(df="df_columns_selected", index_cols="params:index_cols"),
                outputs="df_index_set",
            ),
            node(
                pivot_foreach_label,
                inputs=dict(
                    df="df_index_set",
                    index_column="params:pivot_symbols.index_column",
                    labels_column="params:pivot_symbols.labels_column",
                    values_columns="params:pivot_symbols.values_columns",
                ),
                outputs="df_symbols",
            ),
            node(
                convert_ddf_to_xarray,
                inputs=dict(
                    df_symbols="df_symbols",
                    dask_chunk_size="params:dask.chunksize",
                ),
                outputs="symbols_xarray",
            ),
        ],
    )
