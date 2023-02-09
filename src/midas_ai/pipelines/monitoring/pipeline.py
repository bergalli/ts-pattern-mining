"""
This is a boilerplate pipeline 'plots'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_mlflow_metadata,
    set_mlflow_tracking,
    create_run_uuid,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                create_mlflow_metadata,
                inputs=dict(
                    data_split_rules="params:data_split_rules",
                ),
                outputs=["experiment_name", "experiment_tags"],
                name="mlflow_begin",
            ),
            node(
                set_mlflow_tracking,
                dict(
                    mlflow_tracking_uri="params:mlflow.tracking_uri",
                    mlflow_registry_uri="params:mlflow.registry_uri",
                    mlflow_tracking_user_secret="params:mlflow.tracking_user_secret",
                    mlflow_tracking_password_secret="params:mlflow.tracking_password_secret",
                ),
                outputs=None,
            ),
            node(
                create_run_uuid,
                inputs=dict(run_uuid="params:run_uuid"),
                outputs="run_uuid",
            ),
        ],
        outputs={"experiment_name", "experiment_tags", "run_uuid"},
        parameters={
            "params:data_split_rules",
            "params:run_uuid",
            "params:mlflow.tracking_uri",
            "params:mlflow.registry_uri",
            "params:mlflow.tracking_user_secret",
            "params:mlflow.tracking_password_secret",
        },
        namespace="monitoring",
    )
