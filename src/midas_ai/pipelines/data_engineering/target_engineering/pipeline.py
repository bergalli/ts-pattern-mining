from kedro.pipeline import Pipeline, pipeline, node

from .nodes import select_targets_columns


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                select_targets_columns,
                inputs=dict(symbols_xarray="symbols_xarray",
                            target_metrics=f"params:target.columns"),
                outputs="targets_xr",
                tags=["ml_regression"]
            )
        ]
    ).tag(tags=["targets_engineering"])
