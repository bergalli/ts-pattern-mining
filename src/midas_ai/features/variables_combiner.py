from typing import List, Dict

import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.base import TransformerMixin


def filter_df_keep_bool_dtype_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter all columnes which are not of boolean type.
    :param df:
    :return:
    """
    unique_column_values = df.agg({c: lambda x: set(x.unique().tolist()) for c in df.columns})
    boolean_columns = unique_column_values.map(
        lambda values_set: (values_set == {True, False}) or (values_set == {1, 0})
    )
    df = df.loc[:, boolean_columns]
    return df


def apply_columns_combinations(df, combined_columns):
    """
    From a set of combined (and original) columns, compute the dataframe matching this set, by making combined
    variables as the intersection of columns defined in the combined variable name. Original variables are separated by
    the string "_AND_" in a combined variable name.
    :param df:
    :param combined_columns: list of columns. Can contain both original and combined columns
    :return:
    """
    combined_full = []
    for c1 in combined_columns:
        c1_single_columns = c1.split("_AND_")
        c1_series = df.loc[:, c1_single_columns].all(axis=1)
        c1_series = c1_series.rename(c1)
        combined_full.append(c1_series)
    combined_full = pd.concat(combined_full, axis=1)
    return combined_full


def apply_debias_on_combined_parent_variables(combined_full: pd.DataFrame):
    """
    For each combined or original column, substract the value of its childs present in the combined_full dataframe.

    Ex: substract faces_overall_18_25_AND_product_overall_presence to faces_overall_18_25 and product_overall_presence

    :param combined_full:
    :return:
    """
    for c1 in combined_full.columns:
        combined_childs = [c2 for c2 in combined_full.columns if (c1 in c2) and (c1 != c2)]
        if combined_childs:
            # remove the effect of combined variables child, on their combined/single parent
            combined_full[c1] = (
                (combined_full[c1].astype(float) - combined_full[combined_childs].any(1).values.astype(float)) > 0
            ).astype(float)
    return combined_full


def get_pos_corr_matrix_from_booleans(df):
    """
    Compute correlation between boolean variables, set the diagonal correlation from 1 to 0, set negative
    correlations to 0.
    :param df:
    :return:
    """
    pos_corr_matrix = df.copy().corr(method=lambda a, b: ss.pointbiserialr(a, b)[0]).fillna(0)
    np.fill_diagonal(pos_corr_matrix.values, 0)
    pos_corr_matrix[pos_corr_matrix < 0] = 0
    return pos_corr_matrix


def global_correlation_with_combined(df, combined_columns):
    """

    :param df:
    :param combined_columns:
    :return:
    """
    combined_full = apply_columns_combinations(df, combined_columns)
    combined_full = apply_debias_on_combined_parent_variables(combined_full)
    combined_full = combined_full.drop(combined_full.columns[combined_full.sum() == 0], axis=1)
    return get_pos_corr_matrix_from_booleans(combined_full)


class VariablesCombiner(TransformerMixin):
    """
    Only boolean columns from the input datafarme will be used !
    Output is a dataframe with all colinearities lower than max_corr

    Computing the combined variables is done in 3 steps :
        1. Filter columns that are not booleans, and that contain only 1 value (only zeroes or ones). Initialize the
            set of final columns (comined and original) with the list of original columns (input dataframe columns)
        2. Enter the while loop, and keep looping until conditions are met. Either max colinearity among
         all variables is smaller than self.max_colinearity, and/or all combinations that could lower the maximum
         colinearity of the dataset have been found.
        3. Apply columns combinations to the original dataframe, filter resulting empty columns if any, and saves
         the columns set in self.combined_columns attribute.

    Steps happening in the while loop:
        2a.
        2b.
        2c.

    """

    def __init__(
        self,
        max_comb: int = 2,
        max_colinearity: float = 1.0,
    ):
        """
        :param max_comb
        :param max_colinearity:
        :param cols_not_mix:
        """
        assert 0.0 < max_colinearity <= 1.0
        self.max_correlation = max_colinearity
        self.max_comb = max_comb

        self.combined_columns: List[str] = []

    def fit(self, X: pd.DataFrame):
        """
        Compute a set of combined variables, and substract found combined variables to the original ones when related.
        This operation is done in order to lower the maximum correlation among features of the dataset.
        Found combined columns are then saved in the self.combined_columns attribute.
        :param X:
        :return:
        """
        if (self.max_comb > 1) and (self.max_correlation < 1.0):
            # type all columns as numeric, and drop empty ones.
            ref_df = X.copy().astype(float)
            ref_df = filter_df_keep_bool_dtype_cols(ref_df)
            self.combined_columns = ref_df.columns.tolist()

            # drop uniquely valued columns
            unique_columns_values = ref_df.apply(lambda x: [x.unique().tolist()], axis=0).T.iloc[:, 0]
            single_valued_columns = unique_columns_values[unique_columns_values.map(len) == 1].index
            ref_df = ref_df.drop(single_valued_columns, axis=1)

            while global_correlation_with_combined(ref_df, self.combined_columns).max().max() > self.max_correlation:
                # get absolute correlation between variables
                pos_corr_matrix = global_correlation_with_combined(ref_df, self.combined_columns)
                # filter rows and columns in the correlation matrix, to keep only columns that have been combined
                # less than self.max_comb times, or
                pos_corr_matrix = pos_corr_matrix.loc[
                    pos_corr_matrix.index.map(lambda x: len(x.split("_AND_"))) < self.max_comb,
                    pos_corr_matrix.columns.map(lambda x: len(x.split("_AND_"))) < self.max_comb,
                ]

                # variables to combine together are the top correlated variables
                max_corr_per_variable = pos_corr_matrix.max().max()
                if (len(pos_corr_matrix) == 0) or (max_corr_per_variable <= self.max_correlation):
                    break  # exit while loop , nothing more to do because all variables have reached max_comb.

                to_combine = pos_corr_matrix[pos_corr_matrix == max_corr_per_variable]
                to_combine = pd.DataFrame(
                    np.triu(to_combine),
                    index=to_combine.index,
                    columns=to_combine.columns,
                )
                to_combine[to_combine == 0] = np.nan
                to_combine = to_combine.dropna(how="all", axis=0)

                for colname, to_combine_with in to_combine.iterrows():
                    cols_to_combine = to_combine_with.dropna().index.tolist() + [colname]
                    # combine columns by taking their interesection
                    combined_var_name = "_AND_".join(sorted(cols_to_combine))
                    self.combined_columns.append(combined_var_name)

            ref_df = apply_columns_combinations(ref_df, self.combined_columns)
            ref_df = apply_debias_on_combined_parent_variables(ref_df)
            ref_df = ref_df.drop(ref_df.columns[ref_df.sum() == 0], axis=1)
            self.combined_columns = ref_df.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame):
        if (self.max_comb > 1) and (self.max_correlation < 1.0):
            X_combined = apply_columns_combinations(X, self.combined_columns)
            X_combined = apply_debias_on_combined_parent_variables(X_combined)
            X_combined = X_combined.drop(X_combined.columns[X_combined.sum() == 0], axis=1)
            return X_combined
        else:
            return X
