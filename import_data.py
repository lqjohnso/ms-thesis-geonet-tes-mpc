"""
========================================================================================================================
SCRIPT NAME:    import_data.py
PROJECT:        CHPB-75
AUTHOR(S):      Liam Johnson
AFFILIATION:    Purdue University, CHPB Herrick Labs
DATE CREATED:   8-04-2026
LAST MODIFIED:  10-04-2026
VERSION:        1.2
PYTHON VERSION: 3.14.6
========================================================================================================================
DESCRIPTION: This module contains all functions which import data from API's or datafiles.

DEPENDENCIES:
    - numpy
    - pandas
    - python-openei
    - requests
========================================================================================================================
"""

####################
# Required Imports #
####################

from dataclasses import dataclass
import datetime as dtm
import json
import numpy as np
from openeihttp import Rates
import pandas as pd
from pathlib import Path
import requests
from typing import Literal
from utils import set_timedelta
from utils import get_env_var
import warnings

##########################
# Supporting Dataclasses #
##########################


@dataclass
class RatePlan:
    """A dataclass used exclusively to store information for an energy rate plan.

    Attributes:
        name: The name of the rate plan
        energy_rate: A 1-D vector of energy rate in $/kWh
        demand_rate: A 1-D vector of demand rate in $/kW
        sell_rate: A 1-D vector of sell rate in $/kWh
    """
    # < Declare attributes >
    name: str
    energy_rate: float | np.ndarray
    demand_rate: float | np.ndarray
    sell_rate: float | np.ndarray


#############
# Functions #
#############


async def build_rate_plan(label: str, datetimes: pd.DatetimeIndex | np.datetime64 | dtm.datetime | pd.Timestamp) \
        -> RatePlan:
    """Returns a class instance with name, energy rate, demand rate, and net metering/sell rate for a rate plan
    given the OpenEI label and a datetime index.

    Parameters:
        label: The OpenEI plan label
        datetimes: Single datetime or datetime index

    Output:
        rate_plan: A dataclass with name, energy rate, demand rate, and net metering/sell rate attributes
    """
    # < Validate params >
    assert isinstance(label, str), f"label must be string."
    assert isinstance(datetimes, pd.DatetimeIndex | np.datetime64 | dtm.datetime | pd.Timestamp), \
        f"datetimes must be pd.DatetimeIndex, np.datetime64, dtm.datetime, or pd.Timestamp"

    # < Initialize Rates helper >
    api_key = get_env_var("OPENEI_API_KEY")

    # < Retrieve a specific plan by label (e.g. "539fca56ec12157c50403bf6") >
    try:
        api = Rates(
            api=api_key,
            plan=label
        )
    except AssertionError:
        print(
            "An invalid rate plan label was provided, please ensure your label matches one in the OpenEI database.")

    # < Fetch the rate plan details >
    # Clear cache and force a new API call
    await api.clear_cache()
    await api.update_data()

    # < Get class attributes >
    name = api.rate_name

    if isinstance(datetimes, pd.DatetimeIndex):
        energy_rate = np.empty(len(datetimes))
        demand_rate = np.empty(len(datetimes))
        sell_rate = np.empty(len(datetimes))
        for i, datetime in enumerate(datetimes):
            energy_rate[i] = api.rate(datetime)
            demand_rate[i] = api.demand_rate(datetime)
            sell_rate[i] = api.sell_rate(datetime)
    else:
        energy_rate = api.rate(datetimes)
        demand_rate = api.demand_rate(datetimes)
        sell_rate = api.sell_rate(datetimes)

    # Replace NaN with 0 to make optimization-friendly
    np.nan_to_num(energy_rate, copy=False)
    np.nan_to_num(demand_rate, copy=False)
    np.nan_to_num(sell_rate, copy=False)

    # < Instantiate class >
    return RatePlan(name, energy_rate, demand_rate, sell_rate)


def resample_to_index(df: pd.DataFrame, target_index: pd.DatetimeIndex,
                       downsample_threshold: float = 2.0, inter_type:
        Literal["linear", "time", "index", "values", "nearest", "zero", "slinear", "quadratic", "cubic", "barycentric",
        "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima", "cubicspline", "from_derivatives"] =
              "time", inter_order: int | None = None, limit_direction: Literal["forward", "backward", "both"] = "both")\
        -> pd.DataFrame:
    """Resamples a pandas df to the target indices using mean aggregation if downsampling and inter_type if upsampling.

    Parameters:
        df: The pandas dataframe to be retimed
        target_index: The desired time index for the data
        downsample_threshold: If target's median spacing > this factor*df's median spacing,
            treat as downsampling (aggregate). Otherwise interpolate.
        inter_type: Interpolation method (must be available in pd.df.interpolate)
        inter_order: The order of inter_type if required
        limit_direction: As in pd.DataFrame.interpolate

    Output:
        rate_plan: A dataclass with name, energy rate, demand rate, and net metering/sell rate attributes
    """
    df = df.sort_index()
    target_index = pd.DatetimeIndex(target_index).sort_values()

    src_spacing = df.index.to_series().diff().median()
    tgt_spacing = target_index.to_series().diff().median()

    is_downsampling = (
        pd.notna(src_spacing) and pd.notna(tgt_spacing)
        and tgt_spacing > downsample_threshold*src_spacing
    )

    if is_downsampling:
        # Assign each source point to the largest target timestamp less than or equal to it
        bin_idx = np.searchsorted(target_index.values, df.index.values, side='right') - 1
        valid = bin_idx >= 0

        # Cap the last bin's width to tgt_spacing too, instead of letting it
        # silently absorb everything remaining in df
        if len(target_index) > 0 and pd.notna(tgt_spacing):
            last_bin_end = target_index[-1] + tgt_spacing
            is_last_bin = bin_idx == (len(target_index) - 1)
            valid &= ~(is_last_bin & (pd.DatetimeIndex(df.index.values) >= last_bin_end))

        labels = np.full(len(df), np.datetime64('NaT'), dtype='datetime64[ns]')
        labels[valid] = target_index.values[bin_idx[valid]]

        # Mean aggregation
        result = df.groupby(labels).mean()
        # Restore empty with NaN
        result = result.reindex(target_index)
        result = result.interpolate(method=inter_type, order=inter_order, limit_direction=None)  # interior only
        result = result.interpolate(method='linear', limit_direction='both')  # safe edge fill
        return result

    else:
        full_index = df.index.union(target_index)
        combined = df.reindex(full_index)
        interpolated = combined.interpolate(method=inter_type, order=inter_order, limit_direction=limit_direction)
        return interpolated.loc[target_index]


def retime_df(df: pd.DataFrame, dt: pd.DateOffset | pd.Timedelta | str | None = None, t: pd.DatetimeIndex | None = None,
              utc_offset: str | float | pd.Timedelta | pd.TimedeltaIndex | None = None, inter_type:
        Literal["linear", "time", "index", "values", "nearest", "zero", "slinear", "quadratic", "cubic", "barycentric",
        "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima", "cubicspline", "from_derivatives"] =
              "time", inter_order: int | None = None) -> pd.DataFrame:
    """Retimes pandas dataframe with a datetime index and retimes to dt.

    Parameters:
        df: The pandas dataframe to be retimed
        dt: The desired timestep (timedelta) for the data
        t: An array of times for the data in local time
        utc_offset: The column name, value (in h or timedelta), or array corresponding to the UTC offset which must 
            be applied to the data
        inter_type: Interpolation method (must be available in pd.df.interpolate)
        inter_order: The order of inter_type if required

    Output:
        df: The retimed dataframe
    """
    if utc_offset is not None:
        # < Set to local time >
        if isinstance(utc_offset, str):
            df.index = df.index + df[utc_offset].astype('timedelta64[h]')
        elif isinstance(utc_offset, float | pd.Timedelta):
            df.index = df.index + set_timedelta(utc_offset)
        else:
            assert len(df) == len(utc_offset), (f"length of utc_offset must equal that of df, got {len(utc_offset)} "
                                                f"and {len(df)}.")
            df.index = df.index + utc_offset
    # Remove non-numerical data
    df = df.select_dtypes(include=['number'])
    if t is not None:
        df = resample_to_index(df, t)
    elif dt is not None:
        # < Retime to dt >
        df = df.resample(dt).mean().interpolate(method=inter_type, order=inter_order, limit_direction="both")
    return df


def import_timetable_csv(path: str | Path,
                         dt: pd.DateOffset | pd.Timedelta | str | None = None,
                         t: pd.DatetimeIndex | None = None,
                         utc_offset: str | float | pd.Timedelta | pd.TimedeltaIndex | None = None,
                         inter_type: Literal["linear", "time", "index", "values", "nearest", "zero", "slinear",
                         "quadratic", "cubic", "barycentric", "polynomial", "krogh", "piecewise_polynomial", "spline",
                         "pchip", "akima", "cubicspline", "from_derivatives"] = "time",
                         inter_order: int | None = None) -> pd.DataFrame:
    """Imports a .csv file with a datetime index as a pandas dataframe and retimes to dt.

    Parameters:
        path: The path of the .csv file
        dt: The desired timestep (timedelta) for the data
        t: An array of times for the data in local time
        utc_offset: The column name, value (in h or timedelta), or array corresponding to the UTC offset which must
            be applied to the data
        inter_type: Interpolation method (must be available in pd.df.interpolate)
        inter_order: The order of inter_type if required

    Output:
        df: The dataframe containing data from the .csv
    """
    path = Path(path)
    assert path.exists(), f"filepath does not exist: {path}."
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    return retime_df(df, dt, t, utc_offset, inter_type, inter_order)


def import_timetable_csvs(paths: str | Path | list[str] | list[Path],
                          dt: float | str | pd.DateOffset | pd.Timedelta | None = None,
                          t: pd.DatetimeIndex | None = None,
                          utc_offset: str | float | pd.Timedelta | pd.TimedeltaIndex | None = None,
                          inter_type: Literal["linear", "time", "index", "values", "nearest", "zero", "slinear",
                          "quadratic", "cubic", "barycentric", "polynomial", "krogh", "piecewise_polynomial", "spline",
                          "pchip", "akima", "cubicspline", "from_derivatives"] = "time",inter_order: int | None = None,
                          concat_axis: Literal[0, 1, 'index', 'columns'] = 1,
                          concat_join: Literal['inner', 'outer'] = 'outer') -> pd.DataFrame:
    """Imports one or multiple .csv files with a datetime index as a single pandas dataframe and retimes them to dt.

    Parameters:
        paths: The path or list of paths of .csv files
        dt: The desired timestep (timedelta) for the data
        t: An array of times for the data in local time
        utc_offset: The column name, value (in h or timedelta), or array corresponding to the UTC offset which must
            be applied to the data
        inter_type: Interpolation method (must be available in pd.df.interpolate)
        inter_order: The order of inter_type if required
        concat_axis: Concatenation axis for pd.concat {0/'index', 1/'columns'}
        concat_join:Concatenation join method for pd.concat {'inner', 'outer'} which corresponds to union, intersection

    Output:
        df: The dataframe containing data from the .csv
    """

    if isinstance(paths, list):
        dfs = []
        for path in paths:
            dfs.append(import_timetable_csv(path, dt, t, utc_offset, inter_type, inter_order))
        df = pd.concat(dfs, axis=concat_axis, join=concat_join)
    else:
        df = import_timetable_csv(paths, dt, t, utc_offset, inter_type, inter_order)

    return df


def import_weather(location: str, dt: float | pd.Timedelta | None = None, t: pd.DatetimeIndex | None = None,
                   path: str | Path | None = None, params: list[str] | None = None):
    """Imports weather data for a specific location from a .csv file or the Oikolab API depending on if filename is
    given.

    Parameters:
        location: The city for which data is imported
        dt: The desired timestep for the data in timedelta or in h
        t: An array of times for the data in local time
        path: The weather .csv filepath
        params: Parameters to be fetched from the Oikolab API

    Output:
        df: The dataframe containing weather data
    """
    # < Set dt using dt or t.freq >
    dt = set_timedelta(dt, t)
    if path is not None:
        # < Import weather data by .csv file >
        path = Path(path)
        wth_df = import_timetable_csv(path, dt, t, 'utc_offset (hrs)')
        if location.replace(" ", "") not in str(path.name):
            warnings.warn("location specified does not match that in the weather datafile (based on filename).")
    else:
        # < Import weather data by Oikolab API call >
        # Default parameters
        if params is None:
            params = ['temperature', 'relative_humidity', 'heating_degreeday', 'cooling_degreeday', 'wind_speed',
                      'surface_solar_radiation', 'direct_normal_solar_radiation', 'surface_diffuse_solar_radiation',
                      'soil_temperature_level_1', 'soil_temperature_level_2', 'soil_temperature_level_3',
                      'soil_temperature_level_4']
        assert t is not None, (f"t must be specified if filename is not given (or else Oikolab API call "
                               f"cannot be made).")
        # Fetch API key
        api_key = get_env_var("OIKOLAB_API_KEY")
        url = 'https://api.oikolab.com/weather'

        # Request data as JSON
        # Buffer is added to account for UTC offset
        padd = pd.Timedelta(1, 'D')
        form = '%Y-%m-%d'
        t_start = t[0] - padd
        t_end = t[-1] + padd
        wth_req = requests.get(url,
                               params={'param': params,
                                       'location': location,
                                       'start': t_start.strftime(form),
                                       'end': t_end.strftime(form),
                                       'format': 'json'},
                               headers={'api-key': api_key}
                               )
        # Raise exception if API call fails
        wth_req.raise_for_status()

        # Convert to pandas df
        wth_js = json.loads(wth_req.json()['data'])
        wth_df = pd.DataFrame(
            index=pd.to_datetime(wth_js['index'], unit="s"),
            data=wth_js['data'],
            columns=wth_js['columns'],
        )

        # Retime
        wth_df = retime_df(wth_df, dt, t, 'utc_offset (hrs)')
    return wth_df


def import_load(path: str | Path,
                dt: float | None = None,
                t: pd.DatetimeIndex | None = None,
                col: Literal["Real Thermal Load (kW)", "GLM Thermal Load (kW)", "Real Thermal Load with GLM (kW)"] |
                     int | None = None) -> pd.Series:
    """Imports weather data for a specific location from a .csv file or the Oikolab API depending on if filename is
    given.

    Parameters:
        path: The load .csv filepath
        dt: The desired timestep for the data in timedelta or in h
        t: An array of times for the data in local time
        col: The column of the load datafile which will be imported (as number or name)

    Output:
        load_ser: The series of load data
    """

    # Validate params >
    path = Path(path)

    # < Set dt using dt or t.freq >
    dt = set_timedelta(dt, t)

    # < Set defaults >
    if col is None:
        col = 0

    # < Import df and return series of col >
    load_df = import_timetable_csv(path, dt, t)
    if isinstance(col, int):
        col = load_df.columns[col]
    return load_df[col]
