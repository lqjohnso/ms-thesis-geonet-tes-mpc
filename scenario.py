"""
========================================================================================================================
SCRIPT NAME:    scenario.py
PROJECT:        CHPB-75
AUTHOR(S):      Liam Johnson
AFFILIATION:    Purdue University, CHPB Herrick Labs
DATE CREATED:   7-29-2026
LAST MODIFIED:  10-04-2026
VERSION:        1.7
PYTHON VERSION: 3.14.6
========================================================================================================================
DESCRIPTION: This module defines the scenario class used in the simulation.

DEPENDENCIES:
    - numpy
    - pandas
    - python-openei
    - requests
    - scipy
========================================================================================================================
"""

####################
# Required Imports #
####################

import asyncio
from build_models import Models
from copy import deepcopy
from dataclasses import dataclass
from config_loader import instantiate_class_from_dict
from import_data import build_rate_plan
from import_data import import_load
from import_data import import_weather
from import_data import RatePlan
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal
from utils import edit_del_attr_name_cls
from utils import get_env_var
from utils import make_1d
from utils import make_array
from utils import reindex_cls
from utils import set_timedelta
from utils import smoothdata
from utils import upper_fill_matrix

############
# Defaults #
############

default_data_dir = "data"
default_rn_fol = "reneww_data"
default_dc_fol = "dchouse_data"
default_wth_fol = "weather_data"

####################
# Scenario Classes #
####################


@dataclass
class HouseScenario:
    """A dataclass used to store scenario parameters specific to either the ReNEWW or DC House.

    Attributes:
        rate_plan_baseline: An instance of RatePlan used for the NO TES baseline simulation
        rate_plan_other: An instance of RatePlan used for other simulations
        cop_heat_real: 1-D array of real heating COP values
        cop_cool_real: 1-D array of real cooling COP values
        cop_heat_pred: 1-D array of predicted heating COP values
        cop_cool_pred: 1-D array of predicted cooling COP values
        p_hp_max_heat_real: 1-D array of real heating heat pump electrical power values in kW
        p_hp_max_cool_real: 1-D array of real cooling heat pump electrical power values in kW
        p_hp_max_heat_pred: 1-D array of predicted heating heat pump electrical power values in kW
        p_hp_max_cool_pred: 1-D array of predicted cooling heat pump electrical power values in kW
        p_aux_max: 1-D array of maximum auxiliary heat electrical power values in kW
        q_hp_max_heat_real: 1-D array of real heating heat pump thermal capacity values in kW
        q_hp_max_cool_real: 1-D array of real cooling heat pump thermal capacity values in kW
        q_hp_max_heat_pred: 1-D array of predicted heating heat pump thermal capacity values in kW
        q_hp_max_cool_pred: 1-D array of predicted cooling heat pump thermal capacity values in kW
        load_real: 1-D array of real thermal load values in kW
        load_pred: 1-D array of predicted thermal load values in kW
    """
    # < Declare attributes >
    rate_plan_baseline: RatePlan
    rate_plan_other: RatePlan
    cop_heat_real: np.ndarray
    cop_cool_real: np.ndarray
    cop_heat_pred: np.ndarray
    cop_cool_pred: np.ndarray
    p_hp_max_heat_real: np.ndarray
    p_hp_max_cool_real: np.ndarray
    p_hp_max_heat_pred: np.ndarray
    p_hp_max_cool_pred: np.ndarray
    p_aux_max: np.ndarray
    q_hp_max_heat_real: np.ndarray
    q_hp_max_cool_real: np.ndarray
    q_hp_max_heat_pred: np.ndarray
    q_hp_max_cool_pred: np.ndarray
    load_real: np.ndarray
    load_pred: np.ndarray


class Scenario:
    """A class representing the parameters describing the simulation conditions/scenario.

    Attributes:
        dt (pd.Timedelta): The simulation timestep
        t (pandas.DatetimeIndex): A 1-D array of the simulation datetimes
        t_horiz (pd.Timedelta): The MPC time horizon
        k_horiz (int): The MPC time horizon in timesteps
        objective (str): The optimization objective for the simulation
        controllers (list[str]): List of controller names to be simulated
        x_0 (float): The initial tank energy for the simulation
        soc_eta_guess (float): The estimated state of charge for the simulation which informs eta_d_guess, the constant
            estimate of discharging efficiency used to estimate discharging power during a simulation. A low value
            will have cvxpy overestimate the electrical power of discharging, a high value will underpredict
        comfort_weight (float, None): The weight placed on the discomfort felt by occupants as $/kW departure from
            meeting the thermal load
        depl_weight (float, "auto", None): The weight on the depleted tank energy relative to the initial state in an
            MPC time horizon, which is a scaling factor applied to the mean energy rate during the horizon. Setting to
            "auto" will value the thermal energy depletion based on the GSHP COP at the end of the horizon
        term_weight (float, None): The weight on the terminal cost for stored energy in the tank which is scaling
            applied to the final minus initial simulation state squared in $/(kWh^2)
        disch_viol_weight (float): The weight on the slack variables for qd_rn and qd_dc which strongly discourage
            violating discharging constraints qd[heating] >= 0 qd[cooling] <= 0
        elec_cost_thres (float): The hourly cost in $ for baseline operation used as a threshold for assuming baseline 
            operation in MPC simulations. If the scenario seen by solve_cvx_simplified has an hourly cost below 
            this threshold, baseline (no TES) operation will be automatically assumed
        solve_cvx_function (str): The function in solve_cvx.py used to run convex optimization
        solver (str): The cvxpy solver used in solve_cvx.py
        theta_real (np.ndarray): A 1-D array of the simulation real outdoor temperature in degC
        theta_pred (np.ndarray): A 2-D array of the simulation forecasted outdoor temperature in degC
        temp_egl (np.ndarray): A 1-D array of the GSHP entering ground loop temperature in degC
        rn (HouseScenario): A dataclass containing all scenario parameters specific to the ReNEWW House
        dc (HouseScenario): A dataclass containing all scenario parameters specific to the DC House
    """

    @staticmethod
    def __build_forecast(theta_real: list | np.ndarray,
                         t: pd.DatetimeIndex,
                         dt: float | pd.Timedelta,
                         t_frcst: float | pd.Timedelta,
                         t_update: float | pd.Timedelta,
                         t_shift_max: float | pd.Timedelta,
                         t_start: float | str | pd.Timedelta,
                         precision: int,
                         bias_max: float,
                         lead_max: float | pd.Timedelta,
                         smooth_win: float | pd.Timedelta,
                         rng: np.random.Generator | None = None) -> np.ndarray:
        """Builds a 2-D outdoor temperature forecast matrix given the real temperature, timeindex, and forecast
        parameters.

        Parameters:
            theta_real: A 1-D array of the simulation real outdoor temperature in degC
            t: A 1-D array of datetimes corresponding to theta_real
            dt: The forecast timestep/frequency in h or timedelta
            t_frcst: The time in hours or timedelta representing the forecast horizon
            t_update: The time in hours or timedelta representing how often the forecast updates
            t_shift_max: The time in hours or timedelta representing the maximum timeshift of forecast predictions
            t_start: The time in hours or timedelta representing the starting time of forecast predictions/updates
            precision: The decimal precision of forecasts
            bias_max: The maximum bias of forecasts in degC
            lead_max: A time in hours or timedelta representing the lead time at which maximmum bias/shift occurs
            smooth_win: A time in hours or timedelta representing the smoothing window of the forecast
            rng: An instance of the np.random.Generator class
        """

        # < Set default params >
        if rng is None:
            rng = np.random.default_rng()

        # < Condition params >
        dt = set_timedelta(dt, t)
        theta_real = make_1d(theta_real)
        t_frcst = set_timedelta(t_frcst)
        t_update = set_timedelta(t_update)
        t_shift_max = set_timedelta(t_shift_max)
        t_start = set_timedelta(t_start)
        lead_max = set_timedelta(lead_max)
        # Make t_start the earliest time of day when an update occurs
        t_start = t_start - t_update*(t_start // t_update)
        # Create an array where each row is to be a forecast starting at multiples of t_update from t_start and ending
        # after duration t_frcst
        n = len(t)
        theta_pred = np.full((n, n), np.nan)

        # First forecast update on/after t[0]
        t_start = t[0].normalize() + t_start
        t_start_prev = t_start
        while t_start < t[0]:
            t_start_prev = t_start
            k_start_prev = round((t_start_prev - t[0])/dt)
            t_start += t_update

        # Array of forecast update times
        t_updates = pd.date_range(start=t_start, end=t[-1], freq=t_update)
        update_idx = t.get_indexer(t_updates, method="backfill")
        if 0 in update_idx:
            update_at_start = True
        else:
            update_at_start = False
            update_idx = np.insert(update_idx, 0, 0)

        k_frcst = round(t_frcst/dt)
        for i_idx in update_idx:
            if (i_idx != 0) or update_at_start:
                j_idx_end = min(n-1, i_idx + k_frcst)
            else:
                j_idx_end = min(n-1, k_start_prev + k_frcst)
            if i_idx != j_idx_end:
                j_idx = np.arange(i_idx, j_idx_end + 1)
            else:
                j_idx = make_array(j_idx_end)
            nh = len(j_idx)

            # Lead time
            if i_idx != 0:
                t_lead = t[j_idx] - t[i_idx]
            else:
                t_lead = t[j_idx] - t_start_prev

            # Fraction of maximum bias/drift based on lead time
            frac_max = np.minimum(t_lead/lead_max, 1)

            # Time shift
            shift_profile = frac_max*t_shift_max
            theta_shift = np.zeros(nh)

            for j in range(nh):
                k_shift_max = round(shift_profile[j]/dt)
                k_shift_max = np.minimum(k_shift_max, j - 1)
                k_shift_max = np.minimum(k_shift_max, nh - j)
                k_shift_max = np.maximum(0, k_shift_max)

                if k_shift_max == 0:
                    k_shift = 0
                else:
                    k_shift = rng.integers(-k_shift_max, k_shift_max)
                idx_shift = j_idx[j] - k_shift
                idx_shift = max(0, min(n-1, idx_shift))
                theta_shift[j] = theta_real[idx_shift]

            # Bias drift
            bias_trend = bias_max*frac_max*rng.normal()
            drift = rng.normal(size=nh)
            # params chosen to emulate MATLAB smoothdata function
            drift = smoothdata(drift, smooth_win, dt)
            drift_max = max(abs(drift))
            if drift_max > 0:
                # Normalize to 1
                drift = drift / drift_max
            bias = np.multiply(bias_trend, drift)
            theta_pred[i_idx, j_idx] = np.round(theta_shift + bias, precision)
        # Fill forward between updates
        for d, i0 in enumerate(update_idx):
            if i0 < update_idx[-1]:
                i_end = update_idx[d+1]
            else:
                i_end = n
            for i in range(i0, i_end):
                theta_pred[i, i:n] = theta_pred[i0, i:n]
        return theta_pred

    def __build_house_scenario(
            self,
            house: Literal["dc", "rn"],
            models: Models,
            load_real: np.ndarray,
            rate_labels: list[str],
            normalize_rates: bool = False,
            rng: np.random.Generator | None = None) -> HouseScenario:
        """Returns a class instance of HouseScenario which contains all house-specific scenario attributes.

        Parameters:
            house: The house for which the scenario is generated, "rn" or "dc"
            models: An instance of the Models class
            load_real: 1-D array of the real thermal load for the house in kW
            rate_labels: 2 element list of rate plan labels in the form [rate_label_baseline, rate_label_other]
            rng: An instance of the np.random.Generator class

        Output:
            house_scenario: An instance of HouseScenario
        """
        if rng is None:
            rng = np.random.default_rng()
        assert isinstance(house, str), f"house must be string."
        assert house == "rn" or house == "dc", f"house must be \"rn\" or \"dc\"."

        rate_plans = {'rate_plan_baseline': asyncio.run(build_rate_plan(rate_labels[0], self.t)),
                      'rate_plan_other': asyncio.run(build_rate_plan(rate_labels[1], self.t))}

        if normalize_rates:
            # Normalize baseline rate plan to have the same mean as other rate plan
            rate_plans['rate_plan_baseline'].energy_rate = (
                rate_plans['rate_plan_baseline'].energy_rate*(np.mean(rate_plans['rate_plan_other'].energy_rate) /
                                                              np.mean(rate_plans['rate_plan_baseline'].energy_rate)))

        if house == "rn":
            # Emulate using the current temp_egl measurement as the prediction for the rest of sim
            temp_egl_pred = upper_fill_matrix(self.temp_egl)
            cop = models.gshp.calc_heat_pump_performance("cop", self.temp_egl, temp_egl_pred, self.dt, rng)
            p_hp_max = models.gshp.calc_heat_pump_performance("p_hp_max", self.temp_egl, temp_egl_pred, self.dt, rng)
            p_aux_max = {"p_aux_max": np.repeat(models.gshp.p_aux_max, len(self.t))}
            q_hp_max = models.gshp.calc_heat_pump_performance("q_hp_max", self.temp_egl, temp_egl_pred, self.dt, rng)
            load_pred = models.load_rn.calc_load_pred(self.theta_pred, self.dt)
        else:
            cop = models.ashp.calc_heat_pump_performance("cop", self.theta_real, self.theta_pred, self.dt, rng)
            p_hp_max = models.ashp.calc_heat_pump_performance("p_hp_max", self.theta_real, self.theta_pred,
                                                              self.dt, rng)
            p_aux_max = {"p_aux_max": np.repeat(models.ashp.p_aux_max, len(self.t))}
            q_hp_max = models.ashp.calc_heat_pump_performance("q_hp_max", self.theta_real, self.theta_pred,
                                                              self.dt, rng)
            load_pred = models.load_dc.calc_load_pred(self.theta_pred, self.dt)

        load = {"load_pred": load_pred, "load_real": load_real}

        return instantiate_class_from_dict(HouseScenario, rate_plans | cop | p_hp_max | p_aux_max | q_hp_max | load)

    def __init__(
            self, dt: float | pd.Timedelta,
            t_span: list[str],
            t_horiz: float | pd.Timedelta,
            location: str,
            objective: Literal["Combined Price Arbitrage", "Separate Price Arbitrage"],
            controllers: list[str],
            rnload_csv: str | Path,
            dcload_csv: str | Path,
            models: Models,
            temp_egl_map: float | list | np.ndarray,
            t_sim: float | pd.Timedelta | None = None,
            load_perc: float | None = None,
            soc_0: float | None = None,
            soc_eta_guess: float | None = None,
            comfort_weight: float | None = None,
            depl_weight: float | Literal["auto"] | None = None,
            term_weight: float | None = None,
            disch_viol_weight: float | None = None,
            elec_cost_thres: float | None = None,
            solve_cvx_function: Literal['solve_cvx_scp', 'solve_cvx_simplified', 'solve_cvx_mixedint'] | None = None,
            solver: str | None = None,
            rnload_col: str | int | None = None,
            dcload_col: str | int | None = None,
            weather_csv: str | Path | None = None,
            rate_label_baseline: str | None = None,
            rate_label_other: str | None = None,
            rate_label_rn_baseline: str | None = None,
            rate_label_rn_other: str | None = None,
            rate_label_dc_baseline: str | None = None,
            rate_label_dc_other: str | None = None,
            normalize_rates: bool | None = None,
            t_frcst: float | pd.Timedelta | None = None,
            t_update: float | pd.Timedelta | None = None,
            t_shift_max: float | pd.Timedelta | None = None,
            t_start: float | str | pd.Timedelta | None = None,
            precision: int | None = None,
            bias_max: float | pd.Timedelta | None = None,
            lead_max: float | pd.Timedelta | None = None,
            smooth_win: float | pd.Timedelta | None = None,
            seed: int | None = None,):
        """Initialize attributes of the Scenario class.
    
        Parameters:
            dt: The simulation timestep in h or as timedelta
            t_span: A two-element list containing the start and end datetimes considered in the simulation
            t_horiz: The MPC time horizon in h
            location: The city name in which the simulation takes place
            objective: The optimization objective for the simulation
            controllers: The list of controllers to be simulated
            rnload_csv: The .csv datafile containing load data for the ReNEWW House in kW
            rnload_col: The column in the rnload_csv datafile to use in the simulation (as number or name)
            dcload_csv: The .csv datafile containing load data for the DC House in kW
            dcload_col: The column in the dcload_csv datafile to use in the simulation (as number or name)
            models: An instance of the Models class created in __main__.py
            temp_egl_map: 1-D array containing the coefficients of a polynomial model for the GSHP entering ground loop
                temperature as a function of Oikolab soil temperature
            t_sim: The simulation time in hours
            load_perc: The percentile total (DC + RN) thermal load which occurs halfway through the simulation
            soc_0: The initial state of charge of the tank from -1 to 1 where negative is cooling potential and positive
                is heating
            soc_eta_guess: The estimated state of charge for the simulation which informs eta_d_guess, the constant
                estimate of discharging efficiency used to estimate discharging power during a simulation. A low value
                will have cvxpy overestimate the electrical power of discharging, a high value will underpredict
            comfort_weight: The weight placed on the discomfort felt by occupants as $/kW departure from meeting the
                thermal load
            depl_weight: The weight on the depleted tank energy relative to the initial state in an MPC time horizon,
                which is a scaling factor applied to the mean energy rate during the horizon.  Setting to  "auto" will
                value the thermal energy depletion based on the GSHP COP at the end of the horizon
            term_weight (float, None): The weight on the terminal cost for stored energy in the tank which is scaling
                applied to the final minus initial simulation state squared in $/(kWh^2)
            disch_viol_weight: The weight on the slack variables for qd_rn and qd_dc which strongly discourage violating
                discharging constraints qd[heating] >= 0 qd[cooling] <= 0
            elec_cost_thres: The hourly cost in $ for baseline operation used as a threshold for assuming baseline 
                operation in MPC simulations. If the scenario seen by solve_cvx_simplified has an hourly cost below 
                this threshold, baseline (no TES) operation will be automatically assumed
            solve_cvx_function: The function in solve_cvx.py used to run convex optimization
            solver: The cvxpy solver used in solve_cvx.py
            weather_csv: The .csv file containing Okiolab weather data for the location chosen in the simulation
            rate_label_baseline: The OpenEI rate label for the plan used in the baseline simulations
            rate_label_other: The OpenEI rate label for the plan used in other simulations
            rate_label_rn_baseline: The OpenEI rate label for the plan used for the ReNEWW House in the baseline
                simulations
            rate_label_rn_other: The OpenEI rate label for the plan used for the ReNEWW House in other simulations
            rate_label_dc_baseline: The OpenEI rate label for the plan used for the DC House in the baseline
                simulations
            normalize_rates: Whether to normalize the mean of the energy rate for the baseline plan to the mean of the
                other plan
            rate_label_dc_other: The OpenEI rate label for the plan used for the DC House in other simulations
            t_frcst: The time in hours or timedelta representing the forecast horizon
            t_update: The time in hours or timedelta representing how often the forecast updates
            t_shift_max: The time in hours or timedelta representing the maximum timeshift of forecast predictions
            t_start: The time in hours or timedelta representing the starting time of forecast predictions/updates
            precision: The decimal precision of forecasts
            bias_max: The maximum bias of forecasts in degC
            lead_max: A time in hours or timedelta representing the lead time at which maximum bias/shift occurs
            smooth_win: A time in hours or timedelta representing the smoothing window of the forecast
            seed: The seed chosen for random number generation
        """
        # < Set default params >
        if load_perc is None:
            load_perc = 100
        if soc_eta_guess is None:
            soc_eta_guess = 0.5
        if comfort_weight is None:
            comfort_weight = 0.0
        if disch_viol_weight is None:
            disch_viol_weight = 0.3
        if elec_cost_thres is None:
            elec_cost_thres = 0
        if solve_cvx_function is None:
            solve_cvx_function = 'solve_cvx_simplified'
        if solver is None:
            if solve_cvx_function == 'solve_cvx_mixedint':
                solver = 'GUROBI'
            else:
                solver = 'CLARABEL'
        if rate_label_baseline is not None:
            if rate_label_rn_baseline is None:
                rate_label_rn_baseline = rate_label_baseline
            if rate_label_dc_baseline is None:
                rate_label_dc_baseline = rate_label_baseline
        if rate_label_other is not None:
            if rate_label_rn_other is None:
                rate_label_rn_other = rate_label_other
            if rate_label_dc_other is None:
                rate_label_dc_other = rate_label_other
        if t_frcst is None:
            t_frcst = 48
        if t_update is None:
            t_update = 4
        if t_shift_max is None:
            t_shift_max = 3
        if t_start is None:
            t_start = "00:00"
        if precision is None:
            precision = 1
        if bias_max is None:
            bias_max = 2.5
        if lead_max is None:
            lead_max = 48
        if smooth_win is None:
            smooth_win = 120

        # < Condition params >
        if controllers is str:
            controllers = [controllers]
        dt = set_timedelta(round(dt*60)/60)
        t_sim = set_timedelta(t_sim)
        soc_eta_guess = abs(soc_eta_guess)
        temp_egl_map = make_1d(temp_egl_map)
        t_horiz = set_timedelta(t_horiz)
        t_frcst = set_timedelta(t_frcst)
        t_update = set_timedelta(t_update)

        # < Validate params >
        assert t_horiz <= t_frcst - t_update, (f"t_horiz must be less than or equal to t_frcst - t_update, got "
                                               f"{t_horiz} and {t_frcst - t_update}.")
        controllers_accepted = ["Omniscient Optimal", "No TES Baseline", "RBC TES Baseline", "Causal MPC"]
        for i, controller in enumerate(controllers):
            assert controller in controllers_accepted, (f"\"{controller}\" in controllers is not accepted, the list of "
                                                        f"accepted controllers is: {', '.join(controllers_accepted)}.")
        if load_perc is not None:
            assert (load_perc >= 0) and (load_perc <= 100), (f"load_perc must be a percentage between 0 and 100, got "
                                                             f"{load_perc}.")
        if soc_0 is not None:
            assert -1 <= soc_0 <= 1, f"soc_0 must be between -1 and 1 (inclusive), got {soc_0}."
        assert comfort_weight >= 0, f"comfort_weight must be nonnegative, got {comfort_weight}."
        if term_weight is not None:
            assert term_weight >= 0, f"term_weight must be nonnegative, got {term_weight}."
        assert disch_viol_weight >= 0, f"v must be nonnegative, got {disch_viol_weight}."
        assert (rate_label_rn_baseline is not None) and (rate_label_dc_baseline is not None), \
            (f"rate_label_baseline must be specified unless both rate_label_rn_baseline and rate_label_dc_baseline "
             f"are given.")
        assert (rate_label_rn_other is not None) and (rate_label_dc_other is not None), \
            f"rate_label_other must be specified unless both rate_label_rn_other and rate_label_dc_other are given."

        # < Initialize rng >
        rng = np.random.default_rng(seed)

        # < Calculate class attributes >
        t = pd.date_range(start=t_span[0], end=t_span[1], freq=dt)
        k_horiz = round(t_horiz/dt)
        if soc_0 is not None:
            if soc_0 > 0:
                x_0 = soc_0*models.tank_real.x_max
            else:
                x_0 = -soc_0*models.tank_real.x_min
        else:
            x_0 = None

        # Get datafile folder locations from .env but fall back to defaults if not present
        curr_directory = Path.cwd()
        data_dir = curr_directory / Path(get_env_var("DATA_DIR", default_data_dir))
        rn_dir = data_dir / get_env_var("RN_FOL", default_rn_fol)
        dc_dir = data_dir / get_env_var("DC_FOL", default_dc_fol)
        wth_dir = data_dir / get_env_var("_WTH_FOL", default_wth_fol)

        # Import weather
        wth_df = import_weather(location, dt, t, wth_dir / weather_csv, ['temperature', 'soil_temperature_level_3'])
        theta_real = wth_df['temperature (degC)']
        temp_soil = wth_df['soil_temperature_level_3 (degC)']

        # Import Thermal loads
        rn_load = import_load(rn_dir / rnload_csv, dt, t, rnload_col)
        dc_load = import_load(dc_dir / dcload_csv, dt, t, dcload_col)

        # Find new t if t_sim is specified
        if t_sim is not None:
            assert t_sim < (t[-1] - t[0]), (f"t_sim must be less than the t_span, got {t_sim} and {t[-1] - t[0]} "
                                            f"respectively. ")
            # Absolute value total thermal load
            load_tot_abs = (rn_load + dc_load).abs()
            # Index where percentile occurs
            perc_idx = (load_tot_abs - load_tot_abs.quantile(load_perc/100)).abs().idxmin()
            # Desired starting index
            start_idx = perc_idx - t_sim/2
            end_idx = perc_idx + t_sim/2
            if start_idx <= load_tot_abs.index[0]:
                # < Cannot center load >
                start_idx = load_tot_abs.index[0]
                # Gets closest to t_sim
                end_idx = load_tot_abs.index[np.abs(load_tot_abs.index - (start_idx + t_sim)).argmin()]
            elif end_idx >= load_tot_abs.index[-1]:
                # < Cannot center load >
                end_idx = load_tot_abs.index[-1]
                start_idx = load_tot_abs.index[np.abs(load_tot_abs.index - (end_idx - t_sim)).argmin()]
            else:
                # < Can center load >
                start_idx = load_tot_abs.index[np.abs((load_tot_abs.index - start_idx)).argmin()]
                end_idx = load_tot_abs.index[np.abs(load_tot_abs.index - (start_idx + t_sim)).argmin()]
            t = pd.date_range(start=start_idx, end=end_idx, freq=dt)

        # Ensure datetimes of all attributes match and convert series to np.ndarray
        theta_real = theta_real[t].to_numpy()
        temp_soil = temp_soil[t].to_numpy()
        load_rn_real = rn_load.loc[t].to_numpy()
        load_dc_real = dc_load.loc[t].to_numpy()

        # Build forecasts
        theta_pred = self.__build_forecast(theta_real, t, dt, t_frcst, t_update, t_shift_max, t_start, precision,
                                           bias_max, lead_max, smooth_win, rng)

        # Use curve fit to determine GSHP entering ground loop temperature
        temp_egl = np.polyval(temp_egl_map, temp_soil)

        # < Assign class attributes >
        self.dt = dt
        self.t = t
        self.t_horiz = t_horiz
        self.k_horiz = k_horiz
        self.objective = objective
        self.controllers = controllers
        self.x_0 = x_0
        self.soc_eta_guess = soc_eta_guess
        self.comfort_weight = comfort_weight
        self.depl_weight = depl_weight
        self.term_weight = term_weight
        self.disch_viol_weight = disch_viol_weight
        self.elec_cost_thres = elec_cost_thres
        self.solve_cvx_function = solve_cvx_function
        self.solver = solver
        self.theta_real = theta_real
        self.theta_pred = theta_pred
        self.temp_egl = temp_egl
        self.rn = self.__build_house_scenario("rn", models, load_rn_real,
                                              [rate_label_rn_baseline, rate_label_rn_other],
                                              normalize_rates, rng)
        self.dc = self.__build_house_scenario("dc", models, load_dc_real,
                                              [rate_label_dc_baseline, rate_label_dc_other],
                                              normalize_rates, rng)

    def reduce(self, controller: Literal["Omniscient Optimal", "No TES Baseline", "RBC TES Baseline", "Causal MPC"]):
        """Reduces attributes to those used in the simulation for "controller" and returns as new class."""
        reduced_scenario = deepcopy(self)
        # Use predicted attributes only in the Causal MPC simulation, otherwise use real
        if controller == "Causal MPC":
            edit_del_attr_name_cls(reduced_scenario, "_real", "_pred")
        else:
            edit_del_attr_name_cls(reduced_scenario, "_pred", "_real")

        # Use rate_plan_baseline in the No TES baseline, otherwise use rate_plan_other
        if controller == "No TES Baseline":
            edit_del_attr_name_cls(reduced_scenario, "_other", "_baseline")
        else:
            edit_del_attr_name_cls(reduced_scenario, "_baseline", "_other")

        # Only store name of controller scenario is reduced for
        reduced_scenario.controllers = controller

        return reduced_scenario

    def horiz(self, start_idx: int | None = None) -> Scenario:
        """Re-indexes all relevant array_like attributes of the Scenario class for the appropriate MPC horizon."""
        horiz_scenario = deepcopy(self)
        if start_idx is not None:
            reindex_cls(horiz_scenario, start_idx, horiz_scenario.k_horiz)

        return horiz_scenario

    def update(self, state):
        """Updates initial state parameters."""
        if state.x is not None:
            self.x_0 = state.x[0]
