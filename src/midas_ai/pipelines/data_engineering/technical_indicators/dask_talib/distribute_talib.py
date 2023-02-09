"""
ref: https://towardsdatascience.com/building-a-comprehensive-set-of-technical-indicators-in-python-for-quantitative-trading-8d98751b5fb
"""
import numpy as np
import talib


def _random_walk_sample(shape, dtype, nrows=100):
    """
    Used to build the meta for mapping dask arrays to talib functions
    Args:
        dtype:
        nrows:

    Returns:

    """
    step_set = [-1, 0, 1]
    origin = np.zeros((1, *shape[1:]))

    step_shape = (nrows - 1, *shape[1:])
    steps = np.random.choice(a=step_set, size=step_shape)
    path = np.concatenate([origin, steps]).cumsum(0)

    path = path.astype(dtype)
    return path


def simple_moving_average_dask(vector_metric, timeperiod):
    """
    At timestep t_n, with timeperiod T=3, SMA computation uses timesteps [t_n, t_(n-1), t_(n-2)].
    This implies overlap between chunks must be `ovp = T-1` ,
     so at timestep n there are always [t_n, t_(n-1),... t_(n-T-1)] available for computation
    Boundaries are set to 'none', meaning that beginning and end of the array do not have an overlap added.
    This means at the beginning ( until t_(T-1) ), NaN values will appear.

    Args:
        vector_metric:
        timeperiod:

    Returns:

    """
    sma = vector_metric.map_overlap(lambda series: talib.SMA(series, timeperiod),
                                    depth={0: timeperiod - 1},
                                    boundary={0: 'none'},
                                    trim=True,
                                    meta=_random_walk_sample(vector_metric.shape, vector_metric.dtype,
                                                             nrows=timeperiod * 3))
    return sma


# def sma_ratios(df, symbol_suffix, ma_timeperiods, use_volume):
#     # features_df[f'SMA_15_{symbol}'] = talib.SMA(features_df[f'close_{symbol}'], 15)
#     # features_df[f'SMA_ratio_{symbol}'] = features_df[f'SMA_15_{symbol}'] / features_df[f'SMA_5_{symbol}']
#     #
#     # features_df[f'SMA_5_Volume_{symbol}'] = talib.SMA(features_df[f'volume_{symbol}'], 5)
#     # features_df[f'SMA_15_Volume_{symbol}'] = talib.SMA(features_df[f'volume_{symbol}'], 15)
#     # features_df[f'SMA_Volume_ratio_{symbol}'] = (features_df[f'SMA_15_Volume_{symbol}']
#     #                                            / features_df[f'SMA_5_Volume_{symbol}'])
#     return df

def average_true_range_dask(array_hlc, timeperiod):
    """

    Args:
        array_hlc: array_hlc.chunks must be in the shape of ((...), (3,))
         so that the 3 columns high, low, close are given at the same time in map_overlap
        timeperiod:

    Returns:

    """
    atr = array_hlc.map_overlap(lambda array: talib.ATR(high=array[:, 0],
                                                        low=array[:, 1],
                                                        close=array[:, 2],
                                                        timeperiod=timeperiod),
                                depth={0: timeperiod},
                                boundary={0: 'none'},
                                trim=True,
                                drop_axis=1,  # function changes shape of mapped array. Output loses axis 1.
                                meta=_random_walk_sample(array_hlc.shape, array_hlc.dtype, nrows=timeperiod * 3)
                                )

    return atr


def average_directional_movement_index(df, symbol_suffix, admi_timeperiods):
    high, low, close = df[f'high_{symbol_suffix}'], df[f'low_{symbol_suffix}'], df[f'close_{symbol_suffix}']
    for timeperiod in admi_timeperiods:
        df[f'ADMI_{str(timeperiod)}_{symbol_suffix}'] = talib.ADX(high, low, close, timeperiod)

    return df
