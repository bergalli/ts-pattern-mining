"""
This is a boilerplate pipeline 'dataprep__multivariate_ts'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import pivot_foreach_label, convert_ddf_to_xarray

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            pivot_foreach_label,
            inputs=dict(
                df="df_loaded",
                index_column="params:data_ingestion.dataprep.pivot_foreach_label.index_column",
                labels_column="params:data_ingestion.dataprep.pivot_foreach_label.labels_column",
                values_columns="params:data_ingestion.dataprep.pivot_foreach_label.values_columns"),
            outputs="df_symbols",
            tags=["splitting"]
        ),
        node(
            convert_ddf_to_xarray,
            inputs=dict(df_symbols="df_symbols",
                        dask_chunk_size="params:base.job.scheduler.dask.chunk_size"),
            outputs="symbols_xarray",
            tags=["base_metric_node"]
        ),
    ])
