from functools import partial
from typing import Tuple, List

import dask.array as da
import numpy as np
import pandas as pd
import ray
import xarray as xr
from dask import dataframe as dd


@ray.remote
def shift_array(xr_df: xr.Dataset, period: int, fill_value=np.NaN) -> xr.Dataset:
    shifted_data_vars = {}
    for symbol, dv in xr_df.data_vars.items():
        rolled = da.roll(dv.data, period, axis=0)

        if period < 0:
            rolled[period:, ::] = fill_value
        elif period > 0:
            rolled[:period, ::] = fill_value

        shifted_data_vars.update({symbol: (["timestamp", "metric"], rolled)})
    shifted_array_xr = xr.Dataset(data_vars=shifted_data_vars, coords=xr_df.coords)

    return shifted_array_xr


def shift_features(features_xr: xr.Dataset, timesteps: int, dask_chunk_size) -> xr.Dataset:
    remote_arrays = {period: shift_array.remote(features_xr, period) for period in range(timesteps)}
    features_shifts_xr = {period: ray.get(arr) for period, arr in remote_arrays.items()}

    # create new dimension "shift"
    features_shifts_xr = {
        shift_id: arr.to_array("symbol") for shift_id, arr in features_shifts_xr.items()
    }
    features_shifts_xr = xr.Dataset(features_shifts_xr).to_array("shift")
    features_shifts_xr = features_shifts_xr.to_dataset(dim="symbol")
    features_shifts_xr = features_shifts_xr.transpose("timestamp", "shift", "metric")

    return features_shifts_xr


def handle_nans(
    shifted_features_xr: xr.Dataset, targets_xr: xr.Dataset
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Delete all timesteps containing at least 1 nan
    Args:
        shifted_features_xr:
        targets_xr:

    Returns:

    """

    def get_timestamps_nan_mask(arr: xr.Dataset, dims: List[str]) -> da.Array:
        arr = arr.map(da.isnan).any(dim=dims).to_array("symbol").data.any(0)
        return arr

    features_timestamps_nan = get_timestamps_nan_mask(shifted_features_xr, ["shift", "metric"])
    targets_timestamps_nan = get_timestamps_nan_mask(targets_xr, ["metric"])
    anti_nan_mask = ~(features_timestamps_nan | targets_timestamps_nan)

    features_sqs_nancleaned = shifted_features_xr.where(
        shifted_features_xr.timestamp[anti_nan_mask]
    )
    targets_sqs_nancleaned = targets_xr.where(targets_xr.timestamp[anti_nan_mask])

    return features_sqs_nancleaned, targets_sqs_nancleaned


def filter_overlapping_sequences_features(
    features_all_sequences: xr.Dataset, timesteps: int, to_disjoint_sequences: bool
) -> xr.Dataset:
    if to_disjoint_sequences:  # todo
        # truncate beginning of array if number of sequences not a multiple of timesteps
        features_all_sequences = features_all_sequences[
            (len(features_all_sequences) % timesteps) :, ::
        ]

        data_nrows = len(features_all_sequences)
        # start at -1 to get the last element of features_all_sequences
        ids = list(range(-1, data_nrows, timesteps))[1:]
        features_disjoint_sequences = features_all_sequences[ids, ::]

        return features_disjoint_sequences
    else:
        return features_all_sequences


def agg_overlapping_sequences_target(
    targets_all_sequences: xr.Dataset,
    timesteps: int,
    target_agg_fun,
    to_disjoint_sequences: bool,
) -> xr.Dataset:
    if to_disjoint_sequences:  # todo
        # truncate beginning of array if number of sequences not a multiple of timesteps
        targets_all_sequences = targets_all_sequences[len(targets_all_sequences) % timesteps :, :]

        data_nrows = len(targets_all_sequences)
        # generate groups ids then flatten
        grouper = [[str(i)] * timesteps for i, _ in enumerate(range(0, data_nrows, timesteps))]
        grouper = [i for l_i in grouper for i in l_i]
        # temporarily goes from Array to Dataframe space for groupby
        temp_targets_df = dd.from_dask_array(targets_all_sequences.to_array("symbol").data)
        grouper = dd.from_pandas(
            pd.Series(grouper, name="_grouper"), npartitions=targets_all_sequences.npartitions
        )
        temp_targets_df = dd.multi.merge(
            left=temp_targets_df, right=grouper, left_index=True, right_index=True, how="inner"
        )

        agg_fun = partial(getattr(np, target_agg_fun.fun_name), **target_agg_fun.partial_params)
        # groupby on grouper and aggregate
        targets_disjoint_sequences = (
            temp_targets_df
            # .reset_index()
            .groupby("_grouper")
            .aggregate(agg_fun)
            .to_dask_array()
        )

        targets_disjoint_sequences.compute_chunk_sizes()
        return targets_disjoint_sequences
    else:
        return targets_all_sequences
