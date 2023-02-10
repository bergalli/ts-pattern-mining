from typing import Tuple
import dask.dataframe as dd
import pandas as pd
import ray
import xarray as xr
from ray.air.config import ScalingConfig
from ray.train.lightgbm import LightGBMTrainer
import random


def convert_xarray_to_dask(features_array: xr.Dataset, targets_array: xr.Dataset)-> Tuple[dd.DataFrame, dd.DataFrame]:
    X = features_array.to_array("symbol").stack(dict(columns=["symbol", "metric"])).rename("X")
    Y = targets_array.to_array("symbol").stack(dict(columns=["symbol", "metric"])).rename("Y")

    columns_X = ["__".join(keys) for keys in X.columns.data]
    X = dd.from_dask_array(
        X.data,
        meta=pd.DataFrame(
            X.data[[random.randrange(0, len(X)) for _ in range(100)], :].compute(),
            columns=columns_X,
        ),
        columns=columns_X,
    )
    columns_Y = ["__".join(keys)+"_target" for keys in Y.columns.data]
    Y = dd.from_dask_array(
        Y.data,
        meta=pd.DataFrame(
            Y.data[[random.randrange(0, len(Y)) for _ in range(100)], :].compute(),
            columns=columns_Y,
        ),
        columns=columns_Y,
    )
    return X, Y


def make_train_test_val_datasets(X, Y):
    train_datasets = {}
    for symbol_target in Y.columns:
        train_datasets[symbol_target] = ray.data.from_dask(
            dd.merge(X, Y[symbol_target], left_index=True, right_index=True)
        )
    return train_datasets, train_datasets, train_datasets


def train_model(train_datasets):
    for label_column, train_dataset in train_datasets.items():
        trainer = LightGBMTrainer(
            label_column=label_column,
            params={"objective": "regression"},
            scaling_config=ScalingConfig(num_workers=3),
            datasets={"train": train_dataset},
        )
        train_result = trainer.fit()
    return train_result
