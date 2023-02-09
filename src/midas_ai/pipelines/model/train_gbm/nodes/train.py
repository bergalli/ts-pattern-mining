from typing import Tuple

import dask
import dask.dataframe
import mlflow
import numpy as np
import pandas as pd
import ray
import xarray as xr
from lightgbm import LGBMRegressor
from lightgbm_ray import RayParams
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.util.dask import ray_dask_get
from sklearn.metrics import mean_squared_error


def convert_datasets_to_arrays(
        features_ttv_array: xr.Dataset,
        targets_ttv_array: xr.Dataset
) -> Tuple[xr.DataArray, xr.DataArray]:
    X = (features_ttv_array
         .to_array('symbol')
         .stack(dict(columns=["shift", "symbol", "metric"]))
         .rename('X'))
    y = (targets_ttv_array
         .to_array('symbol')
         .stack(dict(columns=["symbol", "metric"]))
         .rename('y'))
    y = y.assign_coords(columns=pd.MultiIndex.from_tuples(
        [(-1, *target_tuple) for target_tuple in y.indexes['columns']],
        names=['shift'] + y.indexes['columns'].names)
    )

    return X, y


def train_final_model_ray(
        X,
        y,
        best_configs
):
    model_trained = best_configs
    return model_trained


@ray.remote
def _xrarray_to_pandas_dataframe(arr: xr.DataArray, flatten_columns_multiindex=True):
    prim_index = pd.Index(arr.indexes['timestamp'], name='timestamp')
    df = pd.DataFrame(arr.data, columns=arr.indexes['columns']).set_index(prim_index)
    if flatten_columns_multiindex:
        df.columns = df.columns.map(list).map(lambda x: '__'.join([str(xi) for xi in x]))
    return df


@ray.remote
class TTVFoldActor:
    """
    Data storage actor
    """

    def __init__(self, X, y, target_colname=None):
        X.data = X.data.rechunk(chunks=(dask.config.get("array.chunk-size"), *X.data.shape[1:]))
        y.data = y.data.rechunk(chunks=(dask.config.get("array.chunk-size"), *X.data.shape[1:]))
        self.X = X.compute(scheduler=ray_dask_get)
        self.y = y.compute(scheduler=ray_dask_get)

        self.target_colname = target_colname

    def set_target_colname(self, target_colname):
        self.target_colname = target_colname
        return self

    def get_X(self, role, fold_id):
        X_fold = self.X.sel(fold_id=fold_id)
        X_role = _xrarray_to_pandas_dataframe.remote(X_fold.sel(role=role))
        return ray.get(X_role)

    def get_y(self, role, fold_id):
        y_fold = (self.y.sel(columns=self.target_colname)  # todo raise issue multiindex collapsing
                  .sel(fold_id=fold_id)
                  # reassign the dimension to dataarray
                  .expand_dims("columns", -1)
                  # retype as pandas multiindex
                  .assign_coords(columns=pd.MultiIndex.from_tuples([self.target_colname],
                                                                   name=self.y.indexes['columns'].names)))
        y_role = _xrarray_to_pandas_dataframe.remote(y_fold.sel(role=role))
        return ray.get(y_role)

    def get_Xy(self, role, fold_id):
        X = self.get_X(role, fold_id)
        y = self.get_y(role, fold_id)
        return X, y


@ray.remote
def train_a_model(model_class, init_params, X_train, y_train):
    model_instance = model_class(**init_params)
    fitted_model = model_instance.fit(X_train, y_train)
    return fitted_model


@ray.remote
def predict_a_model(fitted_model, X_pred):
    y_pred = fitted_model.predict(X_pred)
    return y_pred


@ray.remote
def train_loop_folds(model_class, init_params, data_actor, n_folds):
    trained_models = {}
    for fold_id in range(n_folds):
        X_ref = data_actor.get_X.remote(role='train', fold_id=fold_id)
        y_ref = data_actor.get_y.remote(role='train', fold_id=fold_id)
        fitted_model = train_a_model.remote(
            model_class=model_class,
            init_params=init_params,
            X_train=X_ref,
            y_train=y_ref
        )
        trained_models[fold_id] = fitted_model
    return trained_models


def tune_hyperparams(
        X: xr.DataArray,
        y: xr.DataArray,
        n_folds: int,
        init_config
):
    def tune_reg_kfolds(config: dict, n_folds, data_actor: TTVFoldActor):
        trained_models = ray.get(train_loop_folds.remote(
            model_class=LGBMRegressor,
            init_params=config["params"],
            data_actor=data_actor,
            n_folds=n_folds
        ))
        predictions = {
            fold_id: predict_a_model.remote(trained_models[fold_id],
                                            X_pred=data_actor.get_X.remote('valid', fold_id))
            for fold_id in range(n_folds)
        }
        mse_list = [
            mean_squared_error(
                ray.get(data_actor.get_y.remote('valid', fold_id)),
                ray.get(y_pred_ref))
            for fold_id, y_pred_ref in predictions.items()
        ]
        mean_mse = np.mean(mse_list)

        tune.report(mse=mean_mse, done=True)

    ray_params = RayParams(
        num_actors=2,
        cpus_per_actor=2,
        max_failed_actors=2
    )
    max_depth = 100
    max_iters = 10000
    # lightgbm parameters guide
    # https://neptune.ai/blog/lightgbm-parameters-guide
    a = {"boosting": "gbdt",  # goss

         # number of boosting iterations (trees to build).
         "num_iterations": tune.uniform(1, max_iters),
         "early_stopping_rounds": tune.uniform(1, 0.1*max_iters),

         # max nb of bins/leafs a variable can be split. aim for max 255
         "max_bin": tune.uniform(1, 10),
         # "min_data_in_leaf": 0,
         # "min_sum_hessian_in_leaf": 0,

         "max_depth": tune.randint(1, max_depth),
         "num_leaves": tune.randint(1, 2 ** max_depth),

         # "bagging_fraction": 0,
         # "bagging_freq": 0,

         # specify the percentage of rows used per tree building iteration.
         "subsample": tune.uniform(0.1, 1.0),
         # deals with column sampling, LightGBM will randomly select
         # a subset of features on each iteration (tree).
         "sub_feature": tune.uniform(0.5, 1.0),

         "lambda_l1": tune.quniform(0, 1, 0.005),
         "lambda_l2": tune.quniform(0, 1, 0.005),

         }
    tune_config = {

        "eta": tune.loguniform(1e-4, 1e-1),
        "max_depth": tune.randint(1, 9)
    }
    config = {**init_config, **tune_config}
    config = {"params": config}

    data_actor = TTVFoldActor.remote(X, y)
    # config["data_actor"] = data_actor

    best_configs = {}
    for target_name in y.indexes['columns']:
        data_actor.set_target_colname.remote(target_name)
        mlflow.set_experiment(experiment_name="mixin_example")
        analysis = tune.run(
            tune.with_parameters(
                tune_reg_kfolds,
                n_folds=n_folds,
                data_actor=data_actor
            ),
            config=config,
            metric="mse",
            mode="min",
            num_samples=10,
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri="sqlite_db/tune.db",
                    experiment_name="mixin_example"
                )
            ],
            # Make sure to use the `get_tune_resources` method to set the `resources_per_trial`
            resources_per_trial=ray_params.get_tune_resources(),
            scheduler=ASHAScheduler(),  # generic
            # search_alg=OptunaSearch(),
            stop={"training_iteration": 50
                  # , "mean_accuracy": 0.98
                  },
            fail_fast=True
        )
        print("Best hyperparameters", analysis.best_config)

        best_configs[target_name] = analysis.best_config
    return best_configs
