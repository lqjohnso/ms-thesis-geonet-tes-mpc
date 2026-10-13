"""
========================================================================================================================
SCRIPT NAME:    simulate.py
PROJECT:        CHPB-75
AUTHOR(S):      Liam Johnson
AFFILIATION:    Purdue University, CHPB Herrick Labs
DATE CREATED:   8-16-2026
LAST MODIFIED:  8-09-2026
VERSION:        1.3
PYTHON VERSION: 3.14.6
========================================================================================================================
DESCRIPTION: This module contains the functions which simulate controllers given the models and scenario.

DEPENDENCIES:
    - coolprop
    - cvxpy
    - numpy
    - pandas
    - python-openei
    - requests
    - toml
    - tqdm
    - scipy
========================================================================================================================
"""

####################
# Required Imports #
####################

from build_models import Models
from dynamics import Control
from dynamics import State
import numpy as np
from scenario import Scenario
from simulate_baselines import simulate_no_tes_baseline
import solve_cvx
import sys
import time
from tqdm import tqdm
from typing import Literal
from utils import round_dct_vals
from warnings import warn

########################
# Simulation Functions #
########################


def __warn_solve_simplified(models: Models, tk_type: Literal['real', 'est']):
    """Warn about duty cycle and optimality compromises when using solve_cvx_simplified with differing maximum and/or
    minimum discharging values for each house."""
    tank = getattr(models, 'tank_' + tk_type)
    if (tank.qd_max_rn != tank.qd_max_dc).any():
        warn(f"qd_max_rn is not equal to qd_max_dc for tank_{tk_type}, this will cause inaccurate duty cycle "
             f"constraints when using 'solve_cvx_simplified' as the solving function. Optimality will be "
             f"sacrificed and discharging duty cycle violations may occur.")
    if (tank.qd_min_rn != tank.qd_min_dc).any():
        warn(f"qd_min_rn is not equal to qd_min_dc for tank_{tk_type}, this will cause inaccurate duty cycle "
             f"constraints when using 'solve_cvx_simplified' as the solving function. Optimality will be "
             f"sacrificed and discharging duty cycle violations may occur.")


def calc_peak_demand(c: Control, s: Scenario) -> dict:
    """Calculates maximum on-peak demand for the controller simulation under the rate plans in s, an
    instance of the scenario class modified by Scenario.reduce().

    Parameters:
        c: An instance of the Control class for the simulation
        s: An instance of the Scenario class reduced to the controller with the desired rate plans using
            Scenario.reduce()
    Output:
        peak_demand: A dictionary containing maximum on-peak demand values for each house and the two combined
    """

    # Total power for each house
    p_tot_rn = c.pd_rn + c.pc + c.p_gshp_rn + c.p_gshp_dc + c.p_aux_gshp
    p_tot_dc = c.pd_dc + c.p_ashp + c.p_aux_ashp
    # Total power for both houses
    p_tot = p_tot_rn + p_tot_dc
    # Energy and demand prices for the ReNEWW House
    pi_e_rn = s.rn.rate_plan.energy_rate
    pi_d_rn = s.rn.rate_plan.demand_rate
    # Energy and demand prices for the DC House
    pi_e_dc = s.dc.rate_plan.energy_rate
    pi_d_dc = s.dc.rate_plan.demand_rate

    # < Calculate on-peak demand reduction >
    if len(set(pi_e_rn)) != 1:
        # Identify on-peak by maximum energy price with TOU rate
        peak_idx_rn = np.where(pi_e_rn == max(pi_e_rn))
    else:
        # Identify on-peak by maximum demand charge with flat rate
        peak_idx_rn = np.where(pi_d_rn == max(pi_d_rn))
    if len(set(pi_e_dc)) != 1:
        peak_idx_dc = np.where(pi_e_dc == max(pi_e_dc))
    else:
        peak_idx_dc = np.where(pi_d_dc == max(pi_d_dc))

    # Individual houses
    peak_demand_rn = max(p_tot_rn[peak_idx_rn])
    peak_demand_dc = max(p_tot_dc[peak_idx_dc])

    # Combined
    # Identify on-peak by maximum energy or demand price based on DC and ReNEWW house rate structures
    peak_idx = np.intersect1d(peak_idx_rn, peak_idx_dc)
    if len(peak_idx) == 0:
        # Default to ReNEWW rate if peaks do not coincide
        peak_demand_tot = max(p_tot[peak_idx_rn])
    else:
        peak_demand_tot = max(p_tot[peak_idx])

    return {'peak_demand_rn': peak_demand_rn,
            'peak_demand_dc': peak_demand_dc,
            'peak_demand_tot': peak_demand_tot,
            }


def simulate(controller: Literal["Omniscient Optimal", "No TES Baseline", "RBC TES Baseline", "Causal MPC"],
             models: Models,
             scenario: Scenario) -> dict:
    """Simulates the controller for the scenario and models specified.

    Parameters:
        controller: The name of the controller to simulate
        models: An instance of the Models class
        scenario: An instance of the Scenario class (not reduced with Scenario.reduce())

    Output:
        results: Results of the simulation in dict format
    """

    # Initialize solve time for controller simulation
    start_time = time.time()

    # Reduce scenario to "real" or "pred" attributes and correct energy rate plan based on controller
    controller_scenario = scenario.reduce(controller)
    # Initialize the state with x_0 from scenario
    state = State.initial(scenario)
    # Choose function to solve convex optimization
    solve_cvx_function = getattr(solve_cvx, scenario.solve_cvx_function)
    if controller == "Omniscient Optimal":
        # Assume we have knowledge of real tank dynamics
        tank = models.tank_real
        # Warn about possible duty cycle violations
        if scenario.solve_cvx_function != 'solve_cvx_scp':
            __warn_solve_simplified(models, 'real')
        # Feed optimization the scenario for entirety of simulation time
        # Single CVX optimization produces final results
        state, control = solve_cvx_function(controller_scenario, tank, state.now)
        # Update the scenario initial state parameters
        scenario.update(state)
    elif controller == "Causal MPC":
        if np.isnan(state.now):
            raise AssertionError("No initial tank state was given to the Casual MPC, this may be due to the omniscient "
                                 "optimization failing. Try setting soc_0 manually in the scenario config.")
        # Create blank control class
        control = Control.blank(scenario)
        # Use the estimated tank model for optimization
        tank = models.tank_est
        # Warn about possible duty cycle violations
        if scenario.solve_cvx_function == 'solve_cvx_simplified':
            __warn_solve_simplified(models, 'est')
        n = len(scenario.t)
        for k in tqdm(range(n), desc="Progress", file=sys.stdout):
            # Feed optimization the scenario for the MPC horizon at each timestep
            # Only return control inputs as state will be calculated later
            control_horiz = solve_cvx_function(controller_scenario.horiz(k), tank, state.now)[1]
            # Clip the control inputs so they match the real scenario and save in the control class instance
            control.clip(control_horiz, scenario, models, state)
            # Update the tank state
            state.update(tank, control)
    elif controller == "No TES Baseline":
        state, control = simulate_no_tes_baseline(controller_scenario)
    else:
        raise KeyError(controller)

    # Calculate solve time in s
    solve_time = time.time() - start_time
    print(f"Simulation complete, solve time is {round(solve_time, 2)}s")

    # Calculate the electricity costs
    elec_costs = solve_cvx.calc_elec_cost(controller_scenario, c=control)

    # Calculate discomfort in average kW
    discomf = {'mean_discomf_rn': np.mean(np.abs(control.q_tot_del_rn - scenario.rn.load_real)),
               'mean_discomf_dc': np.mean(np.abs(control.q_tot_del_dc - scenario.dc.load_real))}

    # Record COP
    cop = {'cop_gshp_heat': scenario.rn.cop_heat_real, 'cop_gshp_cool': scenario.rn.cop_cool_real,
           'cop_ashp_heat': scenario.dc.cop_heat_real, 'cop_ashp_cool': scenario.dc.cop_cool_real}

    # Calculate maximum on-peak demand with the rate plans used in the casual MPC simulation
    peak_demand = calc_peak_demand(control, scenario.reduce("Causal MPC"))

    # Save results as dict
    state_dict = state.__dict__
    del state_dict["_State__count"]
    control_dict = control.__dict__
    del control_dict["_Control__count"]
    results = ({'t': scenario.t, 'load_rn': scenario.rn.load_real, 'load_dc': scenario.dc.load_real,
                'theta': scenario.theta_real, "temp_egl": scenario.temp_egl} | state_dict | control_dict | cop |
               elec_costs | peak_demand | discomf | {"solve_time": solve_time})

    # Round to take care of residuals
    return round_dct_vals(results)
