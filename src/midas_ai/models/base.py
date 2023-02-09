import logging
from abc import ABC, abstractmethod
from typing import Any, List, Dict

import mlflow
import numpy as np
import pandas as pd
from ray import tune
from ray.air import session
from ray.air.callbacks.mlflow import MLflowLoggerCallback
from ray.tune import ExperimentAnalysis
from ray.tune.automl import GeneticSearch, DiscreteSpace, SearchSpace
from ray.tune.schedulers import AsyncHyperBandScheduler

# from ray.tune.search.dragonfly import DragonflySearch
# from ray.tune.search.hebo import HEBOSearch
# from ray.tune.search.flaml import BlendSearch
from midas_ai.features.data_store import DataStore

logger = logging.getLogger(__name__)


class ModelWrapper(ABC):
    def __init__(self, model_cls: Any, init_default_kwargs: dict = {}):
        self.model_cls = model_cls
        self.init_default_kwargs = init_default_kwargs

        self.model = None

        self.endog_train = None
        self.exog_train = None
        self.exog_test = None
        self.endog_test = None

    def new_model(self, **extra_init_args):
        """

        :param extra_init_args: ecrase init_default_kwargs
        :return:
        """
        return self.model_cls(**{**self.init_default_kwargs, **extra_init_args})

    def fit(self, endog: pd.DataFrame, exog: pd.DataFrame, **fit_params):
        self.endog_train = endog
        self.exog_train = exog
        return self._fit(self.endog_train, self.exog_train, **fit_params)

    @abstractmethod
    def _fit(self, endog: pd.DataFrame, exog: pd.DataFrame, **fit_params):
        pass

    @abstractmethod
    def predict(self, exog: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass

    def score(self, endog: pd.DataFrame = None, exog: pd.DataFrame = None, **kwargs) -> float:
        """

        :param endog:
        :param exog:
        :param exog_re:
        :return: score to maximize
        """
        if (endog is not None) and (exog is not None):
            self.exog_test = exog
            self.endog_test = endog

        return self._score(self.endog_test, self.exog_test, **kwargs)

    @abstractmethod
    def _score(self, endog: pd.DataFrame, exog: pd.DataFrame, **kwargs) -> float:
        """

        :param endog:
        :param exog:
        :param exog_re:
        :return: score to maximize
        """
        pass

    @abstractmethod
    def get_coefficients(self):
        pass


class RayModelOptimizer(ModelWrapper):
    OPTIMIZE_MODE = "max"

    def _fit(self, endog: pd.DataFrame, exog: pd.DataFrame, **fit_params):
        raise NotImplementedError

    def predict(self, exog: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def _score(self, endog, exog, **kwargs) -> float:
        raise NotImplementedError

    def get_coefficients(self):
        raise NotImplementedError

    def optimize_features_selection(self, **kwargs) -> ExperimentAnalysis:
        return self._score_optimize_features_selection(
            **kwargs,
            use_mlflow=False,
        )

    def _score_optimize_features_selection(
        self,
        metric: str,
        data_store: DataStore,
        colnames_search_space: List[str],
        fixed_colnames: List[str],
        n_folds: int,
        experiment_name: str,
        experiment_tags: dict,
        run_uuid: str,
        use_mlflow: bool,
        population_size: int,
        population_decay: float,
        max_generation: int,  # todo number_of_iterations
        endog_test,
        exog_test,
    ) -> ExperimentAnalysis:

        optimize_func = tune.with_parameters(
            trainable=self._optimize_func,
            model_cls=self.model_cls,
            init_default_kwargs=self.init_default_kwargs,
            data_store=data_store,
            n_folds=n_folds,
            metric=metric,
            fixed_colnames=fixed_colnames,
            endog_test=endog_test,
            exog_test=exog_test,
        )

        search_alg = self._get_tuner_genetic_optim(
            colnames_search_space=colnames_search_space,
            metric=metric,
            population_size=population_size,
            max_generation=max_generation,
            population_decay=population_decay,
        )

        if use_mlflow:
            callbacks = [
                MLflowLoggerCallback(
                    tracking_uri=mlflow.get_tracking_uri(),
                    # registry_uri = mlflow.get_registry_uri(),
                    experiment_name=experiment_name,
                    tags={
                        **experiment_tags,
                        **dict(
                            run_uuid=run_uuid,
                        ),
                    },
                    save_artifact=False,
                )
            ]
        else:
            callbacks = []

        return tune.run(
            optimize_func,
            metric=metric,
            mode=self.OPTIMIZE_MODE,
            resources_per_trial={"cpu": 1},
            name=experiment_name,
            search_alg=search_alg,
            scheduler=AsyncHyperBandScheduler(),
            callbacks=callbacks,
        )

    @classmethod
    def _optimize_func(
        cls,
        config: Dict[str, bool],
        model_cls: Any,
        init_default_kwargs: dict,
        data_store: DataStore,
        n_folds: int,
        metric: str,
        fixed_colnames: List[str],
        endog_test,
        exog_test,
    ):
        """

        :param config: must be the first parameter defined in the func (ray syntax)
        :param model_cls:
        :param init_default_kwargs:
        :param data_store:
        :param n_folds:
        :param metric:
        :param fixed_colnames:
        :return:
        """
        subset_features = [colname for colname, activated in config.items() if activated]

        train_valid_folds = data_store.get_arrays_splits(n_folds=n_folds)

        loop_perfs, loop_perfs_test = [], []
        for train_valid in train_valid_folds:
            train_arrays = train_valid["train"]
            valid_arrays = train_valid["valid"]

            # subset features in arrays where
            train_arrays = {
                array_name: (
                    array.loc[
                        :,
                        array.columns.intersection(subset_features).tolist()
                        + array.columns.intersection(fixed_colnames).tolist(),
                    ]
                )
                for array_name, array in train_arrays.items()
            }
            valid_arrays = {
                array_name: (
                    array.loc[
                        :,
                        array.columns.intersection(subset_features).tolist()
                        + array.columns.intersection(fixed_colnames).tolist(),
                    ]
                )
                for array_name, array in valid_arrays.items()
            }

            model = cls(model_cls=model_cls, init_default_kwargs=init_default_kwargs)
            model = model.fit(**train_arrays)
            perfs = model.score(**valid_arrays, metric=metric)
            perfs_test = model.score(endog=endog_test, exog=exog_test[["const"] + subset_features], metric=metric)

            loop_perfs.append(perfs)
            loop_perfs_test.append(perfs_test)

        avg_folds_perf = np.mean(loop_perfs)
        avg_folds_perf_test = np.mean(loop_perfs_test)

        session.report(
            {
                metric: avg_folds_perf,
                metric + "_test": avg_folds_perf_test,
            }
        )

    def _get_tuner_genetic_optim(
        self,
        max_generation,
        population_size,
        population_decay,
        colnames_search_space,
        metric,
    ):
        """compute population size and increment max_generation to allow for minimum nomber of final models"""
        assert population_size * (population_decay**max_generation) > 3, "final generation has less than 3 agents"
        # search space : array of booleans of length n_features
        space = SearchSpace(
            {
                DiscreteSpace(colname, [True, False]) if colname not in ["const"]
                # todo gerer column choice constant alleurs qu'ici
                else DiscreteSpace("const", [True])
                for colname in colnames_search_space
            }
        )

        return GeneticSearch(
            space,
            reward_attr=metric,  # always maximized in genetic search
            max_generation=max_generation,
            population_size=population_size,
            population_decay=population_decay,
        )
