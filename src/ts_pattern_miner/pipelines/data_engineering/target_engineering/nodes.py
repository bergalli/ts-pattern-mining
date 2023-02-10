from typing import List

import xarray as xr


def select_targets_columns(
        symbols_xarray: xr.Dataset,
        target_metrics: List[str]
) -> xr.Dataset:
    """
    Same operation as node combine_features_into_dataframe()
    Args:
        symbols_xarray:
        target_metrics:

    Returns:

    """
    targets_xr = symbols_xarray.loc[dict(metric=target_metrics)]
    # targets_xr = targets_xr.assign_coords(dict(role=["target"]))

    return targets_xr
