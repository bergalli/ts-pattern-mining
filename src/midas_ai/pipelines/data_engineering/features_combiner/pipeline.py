"""
This is a boilerplate pipeline 'features_engineering'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import drop_columns_on_level_of_emptyness, combine_variables


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            drop_columns_on_level_of_emptyness,
            inputs=dict(
                df="X",
                extremities_full_or_empty="params:min_col_volume",
            ),
            outputs="X_dense",
        ),
        node(
            combine_variables,
            inputs=dict(
                df="X_dense",
                max_comb="params:variables_combiner.max_comb",
                max_colinearity="params:variables_combiner.max_colinearity",
            ),
            outputs="X_with_combined",
        ),
    ])
