
import pandas as pd
from ts_pattern_miner.features.variables_combiner import VariablesCombiner


def drop_columns_on_level_of_emptyness(df, extremities_full_or_empty: int = 1):
    """
    Filter columns having not enough volume to be analyzed
    :param extremities_full_or_empty: with default value (1), drop columns having (1 or less) or (full or full-1)
    :param df: dataframe of creative features
    :return:
    """
    df = df.astype(float)

    # shape dataframe
    # drop empty or full columns with too much or not enough variance according to extremities_full_or_empty
    # 0 and -1 the two possible missing values
    df = df.drop(
        df.columns[
            (((df == 0).sum() + extremities_full_or_empty) >= df.shape[0])
            | ((len(df) - (df == 0).sum()) <= extremities_full_or_empty)
            | (((df == -1).sum() + extremities_full_or_empty) >= df.shape[0])
            | ((len(df) - (df == -1).sum()) <= extremities_full_or_empty)
        ],
        axis=1,
    )

    return df


def combine_variables(df: pd.DataFrame, max_comb: int, max_colinearity: float) -> pd.DataFrame:
    """
    Combine feature in order to lower the maximum correlation between features in the dataset. See
    VariablesCombiner doc for more details.
    :param max_comb: maximum number of combinations in a variable
    :param max_colinearity: maximum correlation between variables
    :param df: dataset to apply the correlation reduction operation
    :return: dataframe of comined and original (debiased) variables
    """
    vc = VariablesCombiner(
        max_comb=max_comb,
        max_colinearity=max_colinearity,
    )
    df_with_combined = vc.fit_transform(df.astype(float))
    return df_with_combined
