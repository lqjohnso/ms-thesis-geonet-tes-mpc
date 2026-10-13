"""
========================================================================================================================
SCRIPT NAME:    utils.py
PROJECT:        CHPB-75
AUTHOR(S):      Liam Johnson
AFFILIATION:    Purdue University, CHPB Herrick Labs
DATE CREATED:   7-24-2026
LAST MODIFIED:  10-12-2026
VERSION:        1.11
PYTHON VERSION: 3.14.6
========================================================================================================================
DESCRIPTION: This module contains utility functions used in the simulation.

DEPENDENCIES:
    - numpy
    - pandas
    - scipy
========================================================================================================================
"""

####################
# Required Imports #
####################

import datetime as dtm
from copy import deepcopy
import inspect
import numpy as np
from os import getenv
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from typing import TypeVar

############
# Defaults #
############

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#############
# Functions #
#############


def make_array(inpt: float | int | dtm.datetime | np.datetime64 | pd.Timestamp | list | np.ndarray | pd.DatetimeIndex) \
        -> np.ndarray | pd.DatetimeIndex:
    """Makes the input a np.ndarray."""
    assert isinstance(inpt, float | int | dtm.datetime | np.datetime64 | pd.Timestamp | list | np.ndarray |
                      pd.DatetimeIndex), f"input must be float, int, datetime, or array_like."
    if not isinstance(inpt, np.ndarray | pd.DatetimeIndex):
        inpt = np.array([inpt])
    return inpt


def make_1d(inpt: float | int | dtm.datetime | pd.Timestamp | np.datetime64 | list | np.ndarray | pd.DatetimeIndex) \
        -> np.ndarray | pd.DatetimeIndex:
    """Makes the input a 1-D np.ndarray."""
    inpt = make_array(inpt)
    out = inpt.ravel()
    return out


def set_timedelta(dt: float | str | pd.Timedelta | None = None, t: pd.DatetimeIndex | None = None) \
        -> pd.Timedelta | None:
    """Set dt to timedelta using dt as float, hh:mm string or t.freq"""
    if (dt is not None) and ((t is None) or ((t is not None) and (t.freq is None))):
        if isinstance(dt, float | int):
            dt = pd.Timedelta(round(dt*60), "m")
        elif isinstance(dt, str):
            dt = pd.Timedelta(hours=int(dt.split(':')[0]), minutes=int(dt.split(':')[1]))
    elif t is not None:
        dt = pd.Timedelta(t.freq)
    return dt


def mov_mean(arr: np.ndarray, win: int, center: bool = False) -> np.ndarray:
    """Calculates the row-wise moving mean for a np.ndarray while avoiding NaN"""
    # Transpose because pd will calculate column-wise
    df = pd.DataFrame(arr.T)
    df = df.rolling(win, center=center, min_periods=1).mean()
    # Reverse transpose and return as np.ndarray
    return df.to_numpy().T


def clamp(x: float | np.ndarray, max_value: np.ndarray | float = float('inf') ,
          min_value: np.ndarray | float = float('-inf')) -> float| np.ndarray:
    """Clamps the value x within maximum and minimum bounds."""
    return np.maximum(np.minimum(x, max_value), min_value)


def calc_duty_cycle(p: float| np.ndarray, p_max: float| np.ndarray)\
        -> float | np.ndarray:
    """Calculates the duty cycle d of time-averaged power p with maximum value p_max."""
    if isinstance(p, np.ndarray):
        d = np.empty(len(p))
        for i, p_i in enumerate(p):
            if p_i != 0:
                if isinstance(p_max, np.ndarray):
                    d_i = p_i/p_max[i]
                else:
                    d_i = p_i/p_max
                d[i] = clamp(d_i, 0, 1)
            else:
                d[i] = 0
    else:
        if p != 0:
            d = p/p_max
        else:
            d = 0
        d = clamp(d, 1, 0)

    if p_max != 0:
        d = np.divide(p, p_max)
    return d


def __reindex_data(cls: object, attr_name: str, value: pd.Series | pd.DataFrame, start_idx: int, end_idx: int,
                   as_scalar: bool = False):
    """Perform the reindexing step in reindex_cls by row for a series or dataframe."""
    idx_arr = np.arange(start_idx, min(end_idx + 1, len(value)))
    if (len(idx_arr) == 1) and as_scalar:
        idx_arr = idx_arr.item()
    setattr(cls, attr_name, value.iloc[idx_arr])


def __reindex_rows(cls: object, attr_name: str, value: pd.Index | np.ndarray | list | tuple,
                 start_idx: int, end_idx: int, as_scalar: bool = False):
    """Perform the reindexing step in reindex_cls by row."""
    idx_arr = np.arange(start_idx, min(end_idx + 1, len(value)), dtype=int)
    if (len(idx_arr) == 1) and as_scalar:
        idx_arr = idx_arr.item()
    setattr(cls, attr_name, value[idx_arr])


def __reindex_2d(cls: object, attr_name: str, value: np.ndarray, start_idx: int, end_idx: int, as_scalar: bool = False):
    """Perform the reindexing step in reindex_cls for a forecast matrix."""
    idx_arr = np.arange(min(start_idx, value.shape[0]),
                        min(end_idx + 1, value.shape[1]))
    if (len(idx_arr) == 1) and as_scalar:
        idx_arr = idx_arr.item()
        setattr(cls, attr_name, value[idx_arr, idx_arr])
    else:
        setattr(cls, attr_name, value[idx_arr[0], idx_arr])


def is_user_class(obj) -> bool:
    """Checks if a class is user created by checking for source within project root."""
    try:
        src = Path(inspect.getfile(type(obj))).resolve()
    except (TypeError, OSError):
        return False
    return src.is_relative_to(PROJECT_ROOT) and "site-packages" not in src.parts


T = TypeVar("T")


def reindex_cls(cls: T, start_idx: int, n: int = 0, end_idx: int | None = None, as_scalar: bool | None = None,
                exclude: list[str] | set[str] | tuple[str] | tuple[()] = (), _visited=None) -> T:
    """Recursively re-indexes every array-like attribute on `cls` even if nested and leaves other attributes unchanged.

    Parameters:
        cls: A class instance
        start_idx : Starting index (inclusive)
        n : Number of points (end point inclusive)
        end_idx : Ending index (inclusive)
        as_scalar: Whether the result should be returned as a scalar if start_idx = end_idx or only start_idx is given
        exclude : Attributes names to skip
        _visited : set, Tracks object ids already processed to guard against infinite recursion.

    Output:
        cls: The re-indexed class
    """
    if end_idx is None:
        end_idx = start_idx + n
    exclude = set(exclude)
    assert start_idx >= 0, f"start_idx must be greater than or equal to 0"
    if end_idx != -1:
        assert end_idx >= start_idx, f"end_idx must be greater than or equal to start_idx"
    if as_scalar is None:
        as_scalar = False
    if _visited is None:
        _visited = set()
    if id(cls) in _visited:
        return cls
    _visited.add(id(cls))

    # Only proceed if cls is a real class
    if (
        not hasattr(cls, "__dict__")
        or inspect.isclass(cls)
        or inspect.ismodule(cls)
        or inspect.isfunction(cls)
    ):
        return cls

    for attr_name, value in vars(cls).items():
        if attr_name in exclude:
            continue

        if value is None:
            continue

        # Index rows for pd.DataFrame
        if isinstance(value, (pd.Series, pd.DataFrame)):
            __reindex_data(cls, attr_name, value, start_idx, end_idx, as_scalar)

        # By index for pd.Index
        elif isinstance(value, pd.Index):
            __reindex_rows(cls, attr_name, value, start_idx, end_idx, as_scalar)

        elif isinstance(value, np.ndarray):
            if value.ndim == 2:
                # At row start_idx for 2-D np.ndarray (forecast matrix)\
                __reindex_2d(cls, attr_name, value, start_idx, end_idx, as_scalar)
            else:
                # Along axis 0 for np.ndarray that is not 2-D
                __reindex_rows(cls, attr_name, value, start_idx, end_idx, as_scalar)

        # List/tuple
        elif isinstance(value, list | tuple) and not all(isinstance(item, str) for item in value):
            __reindex_rows(cls, attr_name, value, start_idx, end_idx, as_scalar)

        # Utilize recursion for user-created nested class instances by checking if source module is within project root
        if hasattr(value, "__dict__") and not callable(value) and is_user_class(value):
            reindex_cls(value, start_idx, end_idx=end_idx, as_scalar=as_scalar,
                        exclude=exclude, _visited=_visited)

        # Unchanged if not any of the datatypes above

    return cls


def edit_del_attr_name_cls(cls: T, del_str: str, replace_str: str, replace_str_value: str = "",
                           exclude: list[str] | set[str] | tuple[str] | tuple[()] = (), _visited=None) -> T:
    """Recursively deletes every attribute containing del_str and renames every attribute with replace_str on `cls` even
     if nested and leaves other attributes unchanged.

    Parameters:
        cls: A class instance
        del_str: The string which if contained in an attribute name, will cause it to be deleted
        replace_str: The string which if contained in an attribute name, will cause it to be replaced
        replace_str_value: The string which will replace replace_str
        exclude : Attributes names to skip
        _visited : set, Tracks object ids already processed to guard against infinite recursion.

    Output:
        cls: The class with renamed and deleted attributes
    """
    if _visited is None:
        _visited = set()

    obj_id = id(cls)
    if obj_id in _visited:
        return cls
    _visited.add(obj_id)

    # Only proceed if cls is a real class
    if (
        not hasattr(cls, "__dict__")
        or inspect.isclass(cls)
        or inspect.ismodule(cls)
        or inspect.isfunction(cls)
    ):
        return cls

    for attr_name, value in list(vars(cls).items()):
        if attr_name in exclude:
            continue

        if del_str in attr_name:
            delattr(cls, attr_name)
            continue

        if replace_str in attr_name:
            # Create attribute with new name
            new_name = attr_name.replace(replace_str, replace_str_value)
            setattr(cls, new_name, value)
            # Delete old attribute
            delattr(cls, attr_name)

        # Utilize recursion for user-created nested class instances by checking if source module is within project root
        if hasattr(value, "__dict__") and not callable(value) and is_user_class(value):
            edit_del_attr_name_cls(value, del_str, replace_str, replace_str_value, exclude, _visited)

    return cls


def count_init_args(cls):
    """Counts the number of arguments needed to initialize a class."""
    init_method = cls.__init__
    if not callable(init_method):
        return 0
    init_signature = inspect.signature(init_method)

    # Exclude 'self' and return the count of parameters
    return len(init_signature.parameters) - 1


def upper_fill_matrix(arr: np.ndarray) -> np.ndarray:
    """Create a 2-D array representing the prediction horizon available at n,n:end if values in the 1-D input array are
    assumed constant at n."""
    arr = make_1d(arr)
    n = len(arr)
    mask = np.triu(np.ones((n, n), dtype=bool))   # True on/above the diagonal
    return np.where(mask, arr[:, None], np.nan)


def round_dct_vals(dct: dict, prec: int = 4) -> dict:
    """Rounds all float or np.ndarray values in a dictionary."""
    result = {}
    for key, value in dct.items():
        if isinstance(value, dict):
            result[key] = round_dct_vals(value, prec)
        elif isinstance(value, np.ndarray):
            result[key] = np.round(value, prec)
        elif isinstance(value, (float, np.floating)):
            result[key] = round(value, prec)
        else:
            result[key] = value
    return result


def calc_abs_savings(new_val: float, baseline_val: float, prec: int = 2) -> float:
    """Calculates the absolute savings of a new_val compared to its baseline baseline_val and returns with decimal
    precision prec."""
    if (baseline_val == new_val) or (baseline_val == 0):
        return np.nan
    else:
        return round(baseline_val - new_val, prec)


def calc_perc_savings(new_val: float, baseline_val: float, prec: int = 1) -> float:
    """Calculates the percentage savings of a new_val compared to its baseline baseline_val and returns with decimal
    precision prec."""
    if (baseline_val == new_val) or (baseline_val == 0):
        return np.nan
    else:
        return round((baseline_val - new_val)/baseline_val*100, prec)


def forecast_arr_to_vec(arr: np.ndarray) -> np.ndarray:
    """Transforms a 2-D forecast array with lower triangular NaN and finite forecast horizon to a 1-D array filled with
    values updating at every forecast horizon."""
    # Get first row to use as vector
    vec = arr[0, :].copy()

    # Fill in NaN from next starting forecast row and repeat until array is filled
    while np.isnan(vec).any():
        nan_idx = np.where(np.isnan(vec))[0]
        last_row_notnan = np.where(np.logical_not(np.isnan(arr[:, nan_idx[0]])))[0][-1]
        vec[nan_idx] = arr[last_row_notnan, nan_idx].copy()

    return vec


def safe_div(num: float | np.ndarray, denom: float | np.ndarray, val: float | None | np.nan = np.nan)\
        -> float | None | np.nan | np.ndarray:
    """Performs division where the result is np.nan or the specified value when denominator/divisor is zero."""
    # Round denom to prevent numerical errors
    denom = np.round(denom, 7)
    if isinstance(num, float) and isinstance(denom, float):
        return num / denom if denom else val
    else:
        if isinstance(denom, np.ndarray):
            if denom.size == 0:
                return np.empty(0)
            num = np.broadcast_to(num, denom.shape).copy()
            nonzero_idx = denom != 0
            result = np.full(len(denom), val, dtype=float)
            result[nonzero_idx] = np.divide(num[nonzero_idx], denom[nonzero_idx]).copy()
            return result
        else:
            return num / denom if denom else np.full(len(num), val, dtype=float)


def abs_max(a: np.ndarray, axis=None) -> float:
    """Returns the maximum absolute value of a numpy array with its correct sign."""
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)


def pos_dict(dct: dict, exclude: str | list[str] | None = None) -> dict:
    """Returns a dict copy where all values are now positive."""
    dct_copy = deepcopy(dct)
    if isinstance(exclude, str):
        exclude = [exclude]

    for key, value in dct_copy.items():
        if ((exclude is None) or (key not in exclude)) and isinstance(value, float | np.ndarray):
            dct_copy[key] = np.abs(value)
    return dct_copy


def get_env_var(key: str, default_val=None) -> str:
    """Retrieve the value of an environment variable or return a default value when missing and one is given."""
    # Set with default (or None)
    val = default_val

    # Get value of variable
    new_val = getenv(key)
    if new_val is not None:
        val = new_val

    # Don't allow missing key or value without default present
    assert (val is not None) and (isinstance(val, str) and len(val) > 0), \
        f"Please set {key} in the .env file as in .env.example."
    return val


def smoothdata(data: list | np.ndarray, smooth_win: float | pd.Timedelta, dt: float | pd.Timedelta) \
        -> np.array:
    """Applies 1-D Gaussian smoothing to the provided data, emulating MATLAB smoothdata function.

    Parameters:
        data: Input data array
        smooth_win: A time in hours or timedelta representing the smoothing window to be applied
        dt: The data timestep in h or as timedelta

    Output:
        data_smooth: The smoothed data array
    """

    # Validate params
    data = make_array(data)
    dt = set_timedelta(dt)

    # Convert to samples
    smooth_win = round(set_timedelta(smooth_win)/dt)
    
    # Apply filtering
    return gaussian_filter1d(data, smooth_win/5, truncate=2.5, mode='nearest')

