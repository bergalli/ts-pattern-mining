"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from .pipelines import (
    monitoring,
)

from .pipelines import features_engineering, make_sequences, data_load, train_gbm


# warnings.filterwarnings("ignore", category=DeprecationWarning)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    monitoring_pipeline = monitoring.create_pipeline()

    data_load_pipeline = data_load.create_pipeline()
    features_engineering_pipeline = features_engineering.create_pipeline()
    model_pipeline = train_gbm.create_pipeline()
    # disabled for now
    # make_sequences_pipeline = make_sequences.create_pipeline()

    return {
        "__default__": monitoring_pipeline
        + data_load_pipeline
        + features_engineering_pipeline
        + model_pipeline
    }

