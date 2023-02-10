"""
This is a boilerplate pipeline 'dataprep__multivariate_ts'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import pivot_foreach_label, convert_ddf_to_xarray

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([


    ])
