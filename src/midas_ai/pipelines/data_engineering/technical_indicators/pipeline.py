from kedro.pipeline import Pipeline, pipeline, node

from midas_ai.pipelines.data_engineering.technical_indicators.nodes import (
    get_symbols_technical_indicators,
    combine_features_into_dataframe
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                get_symbols_technical_indicators,
                inputs=dict(symbols_xarray="symbols_xarray",
                            ti_conf=f"params:technical_indicators_variations",
                            dask_chunk_size="params:dask.chunk_size"),
                outputs="all_symbols_ti_arrays",
                tags=["technical_indicators"]
            ),
            node(
                combine_features_into_dataframe,
                inputs=dict(symbols_xarray="symbols_xarray",
                            all_symbols_ti_arrays="all_symbols_ti_arrays"),
                outputs="features_xr",
                tags=["aggregator_node"]
            )
        ]
    ).tag(tags=["features_engineering", "technical_indicators"])
