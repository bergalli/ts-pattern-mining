from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def get_kfold_indexes(dataframe, nb_folds,
                      split_col: str = None, strat_cols: list = None, dev_size=0) -> Tuple[pd.Index]:
    """
    For tabular data
    Args:
        dataframe:
        nb_folds:
        split_col:
        strat_cols:
        dev_size:

    Returns:

    """
    """chaque kfold est doublé en inversant test et valid"""
    assert dataframe.index.duplicated().sum() == 0

    if strat_cols is None:
        strat_cols = []

    if split_col is None:  # use index
        if dataframe.index.name is None:
            split_col = 'index'
        else:
            split_col = dataframe.index.name
        dataframe = dataframe.reset_index().set_index(split_col, drop=False)

    if nb_folds > 1:
        if len(strat_cols) > 0:
            dataframe_ref = dataframe[[split_col] + strat_cols].drop_duplicates()
            series_to_split = dataframe_ref[split_col]

            multiclass_vector = multilabel_to_multiclass(multilabels_array=dataframe_ref[strat_cols])
            kf = StratifiedKFold(nb_folds)
            train_testdev_tuples = kf.split(series_to_split, y=multiclass_vector)
        else:
            series_to_split = dataframe[split_col].drop_duplicates()
            kf = KFold(nb_folds, shuffle=True)
            train_testdev_tuples = kf.split(series_to_split)
    else:
        series_to_split = dataframe[split_col].drop_duplicates()
        train_idx, testval_idx = train_test_split(series_to_split, test_size=0.3)
        train_idx_pos = np.where(np.any([np.array(series_to_split) == trix for trix in train_idx], axis=0))[0]
        testdev_idx_pos = np.where(np.any([np.array(series_to_split) == trix for trix in testval_idx], axis=0))[0]
        train_testdev_tuples = [(train_idx_pos, testdev_idx_pos)]

    for (train_idx_pos, testdev_idx_pos) in train_testdev_tuples:
        if dev_size > 0:  # todo stratified entre valid et test
            test_idx_pos, valid_idx_pos = train_test_split(testdev_idx_pos, test_size=dev_size)
        else:
            test_idx_pos, valid_idx_pos = testdev_idx_pos, []
        train_idx = dataframe.index[dataframe[split_col].isin(series_to_split.iloc[train_idx_pos])]
        test_idx = dataframe.index[dataframe[split_col].isin(series_to_split.iloc[test_idx_pos])]
        valid_idx = dataframe.index[dataframe[split_col].isin(series_to_split.iloc[valid_idx_pos])]

        test_val_pair = [(test_idx, valid_idx), (valid_idx, test_idx)]

        for test_idx, valid_idx in test_val_pair:
            yield train_idx, test_idx, valid_idx
