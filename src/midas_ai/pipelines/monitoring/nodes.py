import os
import uuid
from typing import Tuple

import mlflow.tracking


def create_mlflow_metadata(data_split_rules) -> Tuple[str, dict]:
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "exp_name_undefined")
    experiment_tags = {}
    for column_name, filter_rule in data_split_rules.items():
        experiment_tags[column_name] = f'{str(filter_rule["operator"])}_{str(filter_rule["value"]).replace("-", "_")}'
    return experiment_name, experiment_tags


def set_mlflow_tracking(
    mlflow_tracking_uri,
    mlflow_registry_uri,
    mlflow_tracking_user_secret,
    mlflow_tracking_password_secret,
):
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_tracking_user_secret
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_tracking_password_secret

    mlflow.tracking.set_tracking_uri(mlflow_tracking_uri)
    # mlflow.tracking.set_registry_uri(mlflow_registry_uri)


def create_run_uuid(run_uuid) -> str:
    if run_uuid is None:
        return str(uuid.uuid4())
    else:
        return str(run_uuid)


def check_volumes(df):
    print(
        df[["asset_type", "industry_name", "analysis_id"]]
        .drop_duplicates()
        .groupby(["asset_type", "industry_name"])
        .size()
        .sort_values()
        .reset_index()
    )

    print(  # modeles globaux pour detail par verticale p
        df[["asset_type", "objective_family", "analysis_id"]]
        .drop_duplicates()
        .groupby(["asset_type", "objective_family"])
        .size()
        .sort_values()
        .reset_index()
    )

    df[["industry_name", "brand_name"]].drop_duplicates().sort_values(["industry_name", "brand_name"])

