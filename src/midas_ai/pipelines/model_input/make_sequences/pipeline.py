from kedro.pipeline import Pipeline, pipeline, node

from .nodes import (
    shift_features,
    handle_nans,
    filter_overlapping_sequences_features,
    agg_overlapping_sequences_target
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                shift_features,
                inputs=dict(features_xr="features_xr",
                            timesteps="params:make_sequences.timesteps",
                            dask_chunk_size="params:dask.chunk_size"),
                outputs="shifted_features_xr"
            ),
            node(
                handle_nans,
                inputs=dict(shifted_features_xr="shifted_features_xr",
                            targets_xr="targets_xr"),
                outputs=["features_sqs_nancleaned", "targets_sqs_nancleaned"]
            ),
            node(
                filter_overlapping_sequences_features,
                inputs=dict(features_all_sequences="features_sqs_nancleaned",
                            timesteps="params:make_sequences.timesteps",
                            to_disjoint_sequences="params:make_sequences.to_disjoint_sequences"),
                outputs="features_disjoint_sqs"
            ),
            node(
                agg_overlapping_sequences_target,
                inputs=dict(targets_all_sequences="targets_sqs_nancleaned",
                            timesteps="params:make_sequences.timesteps",
                            target_agg_fun="params:target.agg_fun",
                            to_disjoint_sequences="params:make_sequences.to_disjoint_sequences"),
                outputs="target_disjoint_sqs"
            )
        ],
        outputs={
            "features_disjoint_sqs": "features_array",
            "target_disjoint_sqs": "targets_array"
        }
    )