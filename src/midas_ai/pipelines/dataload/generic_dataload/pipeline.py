from kedro.pipeline import Pipeline, pipeline, node


from .nodes import (
    load_dask_df_to_xarray,
    filter_rows_on_configured_rules,
    set_df_index,
    select_columns_from_regex,
)


def create_pipeline(dataset_input, **kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                load_dask_df_to_xarray,
                inputs=dict(df=dataset_input),
                outputs="df_raw",
            ),
            node(
                filter_rows_on_configured_rules,
                inputs=dict(df="df_raw", split_rules="params:data_split_rules"),
                outputs="df_filtered",
            ),
            node(
                select_columns_from_regex,
                inputs=dict(df="df_filtered", regex_colnames="params:regex_colnames"),
                outputs="df_columns_selected"
            ),
            node(
                set_df_index,
                inputs=dict(df="df_columns_selected", index_cols="params:index_cols"),
                outputs="df_index_set",
            ),
        ],
        outputs={"df_index_set": "df_loaded"},
    )
