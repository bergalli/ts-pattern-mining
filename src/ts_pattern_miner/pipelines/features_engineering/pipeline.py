from kedro.pipeline import Pipeline, pipeline, node

from .nodes import (
    get_symbols_technical_indicators,
    shift_features,
    select_targets_columns,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                get_symbols_technical_indicators,
                inputs=dict(
                    symbols_xarray="symbols_xarray",
                    ti_conf=f"params:ti",
                    dask_chunk_size="params:dask.chunksize",
                ),
                outputs="computed_technical_indicators",
            ),
            node(
                shift_features,
                inputs=dict(
                    features_xr="computed_technical_indicators",
                    shift_interval="params:shift_interval",
                ),
                outputs="features_xr",
            ),
            node(
                select_targets_columns,
                inputs=dict(
                    symbols_xarray="symbols_xarray", target_metrics=f"params:target.columns"
                ),
                outputs="targets_xr",
            ),
        ]
    )
