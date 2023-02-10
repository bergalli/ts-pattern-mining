from kedro.pipeline import Pipeline, pipeline, node

from .nodes.train import (
    train_final_model_ray,
    tune_hyperparams,
    convert_datasets_to_arrays
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                convert_datasets_to_arrays,
                inputs=dict(features_ttv_array="features_ttv_array",
                            targets_ttv_array="targets_ttv_array"),
                outputs=["X", "y"]
            ),
            node(
                tune_hyperparams,
                inputs=dict(X="X",
                            y="y",
                            n_folds="params:model_input.make_sequences.ts_ttv_split.n_folds",
                            init_config="params:model_run.run_model.hyperparams"),
                outputs="best_configs"
            ),
            node(
                train_final_model_ray,
                inputs=dict(X="X",
                            y="y",
                            best_configs="best_configs"),
                outputs="model_trained"
            )
        ],
    )
