"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from .pipelines import (
    monitoring,
)
from .pipelines.model import features_optimizer
from .pipelines.model_input import make_sequences, make_data_store
from .pipelines.dataload import generic_dataload
from .pipelines.data_engineering import features_combiner, patterns_profiling, target_engineering, technical_indicators


# warnings.filterwarnings("ignore", category=DeprecationWarning)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    monitoring_pipeline = monitoring.create_pipeline()

    data_load_pipeline = generic_dataload.create_pipeline("daily_crypto_currencies")
    data_engineering_pipeline = new_data_engineering_pipeline()
    features_miner_pipeline = new_features_miner_pipeline()

    return {
        "Forecaster": pipeline([]),
        "Pattern Mining": pipeline([]),
        "Risk Estimation": pipeline([]),
        "RL Trading": pipeline([]),
        "__default__": monitoring_pipeline
        + data_load_pipeline
        + data_engineering_pipeline
        + features_miner_pipeline,
    }


def new_data_engineering_pipeline(**kwargs) -> Pipeline:
    technical_indicators_pipeline = technical_indicators.create_pipeline()
    patterns_profiling_pipeline = patterns_profiling.create_pipeline()
    features_combiner_pipeline = features_combiner.create_pipeline()
    features_engineering_pipeline = (
        technical_indicators_pipeline + patterns_profiling_pipeline + features_combiner_pipeline
    )

    target_engineering_pipeline = target_engineering.create_pipeline()

    return (features_engineering_pipeline + target_engineering_pipeline).tag(["data_engineering"])


def new_features_miner_pipeline(**kwargs) -> Pipeline:
    make_sequences_pipeline = make_sequences.create_pipeline()
    make_data_store_pipeline = make_data_store.create_pipeline()
    features_optimizer_pipeline = features_optimizer.create_pipeline()
    return make_sequences_pipeline + make_data_store_pipeline + features_optimizer_pipeline
