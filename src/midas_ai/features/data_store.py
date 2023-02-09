import logging
from collections import Counter
from functools import partial, reduce
from itertools import combinations
from typing import List, Dict, Optional

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataStore:
    """
    Data handling actor to generate train/validation datasets.
    Manages "calls" made to the data stored on cluster.
    """

    def __init__(
        self,
        kfold_groups: Optional[pd.DataFrame] = None,
        **array_name_2_array: pd.DataFrame,
    ):
        """

        :param kfold_groups: data used only internally, to stratify train_test split
        :param combine_variables:
        :param array_name_2_array: named arrays in pd.DataFrame format.
            Access with  DataStore.get_array_by_name or DataStore.all_arrays
        """
        array_name_2_array["kfold_groups"] = kfold_groups

        assert all(
            isinstance(array, pd.DataFrame) for array in array_name_2_array.values()
        ), "All kwargs must be dataframes"
        for array_name, array in array_name_2_array.items():
            setattr(self, array_name, array)

        self.arrays_names = list(array_name_2_array.keys())

        self._test_all_columns_unique()
        self._test_all_indexes_overlap()

        self._align_dataframes_on_index()
        self._initialize_traintest_split_stratas()

    def _test_all_columns_unique(self):
        """
        Test if no column is a duplicate inside an array or between arrays
        Not a necessity for class' methods to run, but a security to not have duplicate features.
        :return:
        """
        all_columns = [
            col
            for array_colnames in map(
                lambda x: x.columns,
                [getattr(self, array_name) for array_name in self.arrays_names],
            )
            for col in array_colnames
        ]
        if max(Counter(all_columns).values()) > 1:
            raise Exception("duplicate columns between arrays")

    def _test_all_indexes_overlap(self):
        """
        Test if all provided arrays share the same indexes
        :return:
        """
        assert all(
            df1.index.isin(df2.index).all() & df2.index.isin(df1.index).all()
            for df1, df2 in combinations(self.all_arrays, r=2)
        )

    def _align_dataframes_on_index(self):
        common_index = reduce(
            lambda idx1, idx2: idx1.intersection(idx2),
            [arr.index for arr in self.all_arrays],
        )
        for array_name, array in self.all_named_arrays.items():
            setattr(self, array_name, array.loc[common_index, :])

    def _initialize_traintest_split_stratas(self):
        """
        cross join columns of List[str] // will have duplicate indexes
        goal is to have all combinations of lists items in different columns
        :return:
        """
        kfold_groups = self.get_array_by_name("kfold_groups")
        exploded_groups_columns = [
            colvalues.explode().to_frame() for colname, colvalues in kfold_groups.iteritems()
        ]
        cross_joined = pd.DataFrame(
            reduce(
                partial(pd.merge, left_index=True, right_index=True, how="inner"),
                exploded_groups_columns,
            )
        )

        self.kfgroups_to_idx = pd.DataFrame(
            index=pd.MultiIndex.from_frame(cross_joined),
            data=cross_joined.index.values,
            columns=cross_joined.index.names,
        )

    def get_arrays_splits(self, n_folds):
        unique_groups = self.kfgroups_to_idx.index.unique()
        while n_folds > len(unique_groups):
            n_folds -= 1
        else:
            if (n_folds == 1) or (n_folds == 2):
                # raise ModelingError("Cannot do kfolds splits with only 1 group")
                return self._get_train_test_split()
        return self._get_kfolds_splits(n_folds)

    def _get_train_test_split(self) -> List[Dict[str, Dict[str, pd.DataFrame]]]:

        train_data, valid_data = train_test_split(self.kfgroups_to_idx)
        idx = self.get_shared_index()
        train_filter, valid_filter = (
            list(idx.isin(train_data[idx.names].values.ravel())),
            list(idx.isin(valid_data[idx.names].values.ravel())),
        )
        arrays_train, arrays_valid = self.get_train_valid_arrays(train_filter, valid_filter)
        return [
            {
                "train": arrays_train,
                "valid": arrays_valid,
            }
        ]

    def _get_kfolds_splits(self, n_folds: int) -> List[Dict[str, Dict[str, pd.DataFrame]]]:
        """

        :param n_folds:
        :return: a dictionary with train/valid keys, pointing to a list of arrays_name_2_array dictionnaries
            for each
        """
        unique_groups = self.kfgroups_to_idx.index.unique()
        kf = KFold(n_folds)
        train_valid_folds = []
        for groups_train_idx, groups_valid_idx in kf.split(unique_groups):
            groups_train, groups_valid = (
                unique_groups[groups_train_idx],
                unique_groups[groups_valid_idx],
            )

            train_filter, valid_filter = (
                self.kfgroups_to_idx.loc[groups_train.tolist(), :].values.ravel().tolist(),
                self.kfgroups_to_idx.loc[groups_valid.tolist(), :].values.ravel().tolist(),
            )

            arrays_train, arrays_valid = self.get_train_valid_arrays(train_filter, valid_filter)

            train_valid_folds.append(
                {
                    "train": arrays_train,
                    "valid": arrays_valid,
                }
            )

        return train_valid_folds

    def get_train_valid_arrays(self, train_filter: List[bool], valid_filter: List[bool]):
        arrays_full = {  # todo remplacer par self.all_arrays en gerant le if kfold_groups ailleurs
            arr_name: arr
            for arr_name, arr in self.all_named_arrays.items()
            if arr_name != "kfold_groups"
        }
        arrays_train, arrays_valid = (
            {arr_name: arr.loc[train_filter, :] for arr_name, arr in arrays_full.items()},
            {arr_name: arr.loc[valid_filter, :] for arr_name, arr in arrays_full.items()},
        )
        return arrays_train, arrays_valid

    def get_array_by_name(self, arr_name: str) -> pd.DataFrame:
        """
        Return the dataframe named `arr_name`
        :param arr_name:
        :return:
        """
        return getattr(self, arr_name)

    def get_shared_index(self):
        all_indexes = [df.index for df in self.all_arrays]
        return reduce(lambda idx1, idx2: idx1.intersection(idx2), all_indexes)

    @property
    def all_named_arrays(self) -> Dict[str, pd.DataFrame]:
        return {arr_name: self.get_array_by_name(arr_name) for arr_name in self.arrays_names}

    @property
    def all_arrays(self) -> List[pd.DataFrame]:
        """
        Get arrays as a list of dataframes, or a list of (name, df) tuples if return_name=True
        :param with_name:
        :return:
        """
        return [self.get_array_by_name(arr_name) for arr_name in self.arrays_names]

    @staticmethod
    def align_dataframes(X, Y, df_input):
        """
        Align dataframes on index.
        Train dataframes are also re-aligned in the datastore actor
        :param X:
        :param Y:
        :param df_input:
        :return:
        """
        # in case indexes are not aligned
        common_index = reduce(
            lambda idx1, idx2: idx1.intersection(idx2),
            [X.index, Y.index, df_input.reset_index().set_index("analysis_id").index],
        )
        assert len(common_index) > 0, "no matching observations between some or all training arrays"

        X, Y, df_input = (
            X.loc[common_index, :],
            Y.loc[common_index, :],
            df_input.reset_index().set_index("analysis_id").loc[common_index, :].reset_index(),
        )
        return X, Y, df_input
