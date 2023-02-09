import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from ray.tune import ExperimentAnalysis

from midas_ai.features.data_store import DataStore
from midas_ai.models.classification import ClassificationStatsmodels

logger = logging.getLogger("kedro")


def optimize_a_binomial_model(
    data_store: DataStore,
    model_perf_metric: str,
    n_folds: int,
    experiment_name: str,
    experiment_tags: dict,
    run_uuid: str,
    population_size: int,
    population_decay: float,
    max_generation: int,
    endog_test,
    exog_test,
) -> ExperimentAnalysis:
    """
    Feature selection happens here, through an optimization conducted by a genetic search.

    :param data_store: train/valid data
    :param model_perf_metric: metric to be optimized through the search. Can be 'auc', 'acc' (accuracy), or 'f1'
    :param n_folds: number of kfolds to train on
    :param experiment_name: name of experiment in mlflow
    :param experiment_tags: tags common to each run in this mlflow experiment
    :param run_uuid: pipeline run id
    :param population_size: starting population for the genetic search
    :param population_decay: number of individuals to keep after each generation in the genetic search
    :param max_generation: total number of generations for the genetic search
    :param exog_test: provided to the optimizer, but only used as visual information, not used to optimize
    :param endog_test: provided to the optimizer, but only used as visual information, not used to optimize
    :return: optimisation results
    """
    exog_colnames, endog_colnames = (
        data_store.get_array_by_name("exog").columns.tolist(),
        data_store.get_array_by_name("endog").columns.tolist(),
    )

    colnames_search_space, fixed_colnames = [
        c for c in exog_colnames if c != "const"
    ], endog_colnames + ["const"]

    model = binomial_glm = ClassificationStatsmodels(
        model_cls=sm.GLM,
        init_default_kwargs=dict(family=sm.families.Binomial(link=sm.families.links.Logit())),
    )
    optimization_results = model.optimize_features_selection(
        data_store=data_store,
        metric=model_perf_metric,
        colnames_search_space=colnames_search_space,
        fixed_colnames=fixed_colnames,
        n_folds=n_folds,
        population_size=population_size,
        population_decay=population_decay,
        max_generation=max_generation,
        experiment_name=experiment_name,
        experiment_tags=experiment_tags,
        run_uuid=run_uuid,
        endog_test=endog_test,
        exog_test=exog_test,
    )
    return optimization_results


def retrain_best_binomial_model(
    data_store: DataStore,
    optimization_results: ExperimentAnalysis,
    exog_test,
    endog_test,
    model_perf_metric,
) -> ClassificationStatsmodels:
    """

    :param data_store:
    :param optimization_results:
    :param exog_test:
    :param endog_test:
    :param model_perf_metric:
    :return:
    """

    results_df = optimization_results.results_df
    train_test_perfs_isclose = results_df.loc[
        :, [model_perf_metric, model_perf_metric + "_test"]
    ].apply(
        lambda row: np.isclose(row[model_perf_metric], row[model_perf_metric + "_test"], rtol=0.05),
        axis=1,
    )
    valid_models = results_df.loc[train_test_perfs_isclose, :]
    best_config = valid_models[
        valid_models[model_perf_metric] == valid_models[model_perf_metric].max()
    ]
    assert len(best_config) > 0, "no model have an equivalent perf in train and test"
    best_config = best_config.iloc[0, :]
    best_config = best_config.loc[best_config.index.str.startswith("config/")]
    best_config.index = best_config.index.str.replace("config/", "")
    features_selection = [colname for colname, activated in best_config.items() if activated]
    features_selection += ["const"]

    exog = data_store.get_array_by_name("exog")
    endog = data_store.get_array_by_name("endog")

    model = binomial_glm = ClassificationStatsmodels(
        model_cls=sm.GLM,
        init_default_kwargs=dict(family=sm.families.Binomial(link=sm.families.links.Logit())),
    )
    model.fit(
        endog,
        exog.loc[:, features_selection],
    )

    score = model.score(
        endog_test,
        exog_test.loc[:, features_selection],
        metric=model_perf_metric,
    )

    logger.debug(f"Best model score using metric {model_perf_metric} : {score}")

    return model


def extract_glm_coefficients(
    dataprep_table: pd.DataFrame, trained_model: ClassificationStatsmodels
) -> pd.DataFrame:
    """

    :param dataprep_table:
    :param trained_model:
    :return:
    """
    perimeter_columns = ["asset_type", "industry_name", "objective_family"]
    coefficients = trained_model.get_coefficients()

    perimeter_frame = dataprep_table[perimeter_columns]
    perimeter = perimeter_frame.drop_duplicates()
    assert perimeter.shape[0] == 1, "Perimeter has not been correctly defined in conf"

    coefs_frame = coefficients[["variable", "coef", "pvalue"]]
    # repeat the values along index
    coefs_frame = pd.concat(
        [
            coefs_frame,
            pd.DataFrame(
                perimeter.values.tolist(),
                index=coefs_frame.index,
                columns=perimeter_columns,
            ),
        ],
        axis=1,
    )
    return coefs_frame
