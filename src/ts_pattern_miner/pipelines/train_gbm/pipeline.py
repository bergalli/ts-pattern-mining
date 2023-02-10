from kedro.pipeline import Pipeline, pipeline, node

from .nodes import (
    convert_xarray_to_dask
, make_train_test_val_datasets
, train_model
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                convert_xarray_to_dask,
                inputs=dict(features_array="features_xr",
                            targets_array="targets_xr"),
                outputs=["X", "Y"]
            ),
            node(
                make_train_test_val_datasets,
                inputs=dict(X="X",
                            Y="Y",),
                outputs=["train_data", "test_data", "valid_data"]
            ),
            node(
                train_model,
                inputs=dict(train_datasets="train_data"),
                outputs="train_result"
            )
        ],
    )
