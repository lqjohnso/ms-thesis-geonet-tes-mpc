"""
========================================================================================================================
SCRIPT NAME:    solve_cvx.py
PROJECT:        CHPB-75
AUTHOR(S):      Liam Johnson
AFFILIATION:    Purdue University, CHPB Herrick Labs
DATE CREATED:   8-14-2026
LAST MODIFIED:  10-06-2026
VERSION:        1.6
PYTHON VERSION: 3.14.6
========================================================================================================================
DESCRIPTION: This module contains the function which solves a convex optimization problem given the scenario, tank
model, and initial state.

DEPENDENCIES:
    - coolprop
    - cvxpy
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

import cvxpy as cvx
from build_scenario import Scenario
from dynamics import Control
from dynamics import State
from models import Tank
import numpy as np
import pandas as pd
from simulate_baselines import simulate_no_tes_baseline
from utils import abs_max
from utils import clamp
from utils import safe_div
from warnings import warn

##########################
# Optimization Functions #
##########################


def calc_elec_cost(s: Scenario, p_tot_rn: cvx.Variable | cvx.Expression | None = None,
                   p_tot_dc: cvx.Variable | cvx.Expression | None = None,
                   c: Control | None = None) -> cvx.Expression | dict:
    """CVX-friendly function which uses cvxpy atoms to calculate the instantaneous electricity costs that occur over
    the optimization horizon, specifically if that horizon includes multiple months (will also return actual value if a
    full instance of Control is given).

    Parameters:
        s: An instance of Scenario class for the optimization horizon which has been reduced with Scenario.reduce()
        p_tot_rn : A 1-D cvxpy variable/expression of the total power consumed in the ReNEWW house in kW
        p_tot_dc : A 1-D cvxpy variable/expression of the total power consumed in the DC house in kW
        c: A full instance of the Control class for when an actual cost array is to be calculated

    Output:
        elec_cost_obj: A cvxpy expression or dict representing the instantaneous electricity cost in $ used in the
            objective function or all quantitative electricity cost measures.
    """

    assert (((p_tot_rn is not None) and (p_tot_dc is not None)) or
            ((c is not None) and (True not in np.isnan(c.pd_rn)))), \
        f"p_tot_rn and p_tot_dc must be given unless c is specified."

    # dt converted to float h
    dt = round(s.dt.seconds/60)/60
    # Energy and demand prices for the ReNEWW House
    pi_e_rn = s.rn.rate_plan.energy_rate
    pi_d_rn = s.rn.rate_plan.demand_rate
    # Energy and demand prices for the DC House
    pi_e_dc = s.dc.rate_plan.energy_rate
    pi_d_dc = s.dc.rate_plan.demand_rate

    # Total power expressions or arrays
    if (c is not None) and (True not in np.isnan(c.pd_rn)):
        p_tot_rn = c.pd_rn + c.pc + c.p_gshp_rn + c.p_gshp_dc + c.p_aux_gshp
        p_tot_dc = c.pd_dc + c.p_ashp + c.p_aux_ashp
        is_cvx = False
    else:
        is_cvx = True
    p_tot = p_tot_rn + p_tot_dc

    # Electricity cost expressions with energy and demand charges
    energy_cost_rn = cvx.multiply(p_tot_rn, pi_e_rn)*dt
    energy_cost_dc = cvx.multiply(p_tot_dc, pi_e_dc)*dt
    demand_cost_rn = calc_demand_charge_cost(p_tot_rn, s.t, pi_d_rn)
    demand_cost_dc = calc_demand_charge_cost(p_tot_dc, s.t, pi_d_dc)
    demand_cost_tot = calc_demand_charge_cost(p_tot, s.t, pi_d_dc)
    energy_cost_tot = energy_cost_rn + energy_cost_dc
    elec_cost_rn = cvx.sum(energy_cost_rn) + demand_cost_rn
    elec_cost_dc = cvx.sum(energy_cost_dc) + demand_cost_dc
    elec_cost_tot = cvx.sum(energy_cost_tot) + demand_cost_tot
    if s.objective == "Combined Price Arbitrage":
        # Single demand charge
        elec_cost_obj = elec_cost_tot
    elif s.objective == "Separate Price Arbitrage":
        # Separate demand charges per house
        elec_cost_obj = elec_cost_rn + elec_cost_dc
    else:
        raise ValueError(f"Objective \"{s.objective}\" not recognized.")

    # Return either CVX expression or np.ndarray
    if is_cvx:
        return elec_cost_obj
    else:
        return {'elec_cost_obj': round(elec_cost_obj.value, 2),
                'energy_cost_rn': energy_cost_rn.value,
                'energy_cost_dc': energy_cost_dc.value,
                'energy_cost_tot': energy_cost_tot.value,
                'demand_cost_rn': round(demand_cost_rn, 2),
                'demand_cost_dc': round(demand_cost_dc, 2),
                'demand_cost_tot': round(demand_cost_tot, 2),
                }


def calc_demand_charge_cost(p: cvx.Variable | cvx.Expression | np.ndarray, t: pd.DatetimeIndex, pi_d: np.ndarray) \
        -> cvx.Expression | float:
    """CVX-friendly function which uses cvxpy atoms to calculate the demand charges that occur over the optimization
    horizon, specifically if that horizon includes multiple months (will also return actual value if an array of total
    power is given).

    Parameters:
        p: A 1-D cvxpy variable or array of the total power consumed in the house in kW
        t : A 1-D array containing the datetimes for the simulation timesteps
        pi_d: A 1-D array containing the demand charge at each timestep in $/kW

    Output:
        demand_cost: A cvxpy expression representing the total demand charge cost
    """

    # year-month as string for all t
    year_months = t.to_period("M")

    monthly_demand_cost = []
    for year_month in year_months.unique():
        idx = np.where(year_months == year_month)[0]
        demand_cost = cvx.max(cvx.multiply(p[idx], pi_d[idx]))
        monthly_demand_cost.append(demand_cost)

    demand_cost_tot = cvx.sum(monthly_demand_cost)

    if isinstance(p, cvx.Variable | cvx.Expression):
        return demand_cost_tot
    else:
        return demand_cost_tot.value.item()


def calc_duty_cycle_power(st: State, c: Control, tk: Tank, s: Scenario | None = None):
    """Calculate the charging and discharging duty cycle and power for an instance of the Control class."""

    # Charging
    if s is not None:
        c.dc = np.maximum(c.qc/s.rn.p_hp_max_heat/s.rn.cop_heat, c.qc/s.rn.p_hp_max_cool/s.rn.cop_cool)

    # Discharging
    idx_x_heat = st.x > 0
    idx_x_cool = st.x <= 0
    c.dd_rn[idx_x_heat] = safe_div(c.qd_rn[idx_x_heat], c.qd_max_rn[idx_x_heat], 0)
    c.dd_rn[idx_x_cool] = safe_div(c.qd_rn[idx_x_cool], c.qd_min_rn[idx_x_cool], 0)
    c.dd_dc[idx_x_heat] = safe_div(c.qd_dc[idx_x_heat], c.qd_max_dc[idx_x_heat], 0)
    c.dd_dc[idx_x_cool] = safe_div(c.qd_dc[idx_x_cool], c.qd_min_dc[idx_x_cool], 0)
    c.pd_rn = c.dd_rn*tk.pd_max_rn
    c.pd_dc = c.dd_dc*tk.pd_max_dc


def calc_disch_limits(st: State, c: Control):
    """Correct qd_max and qd_min so they reflect the piecewise nature of the true function."""

    # Lower bound should be zero when tank has heating potential, upper bound should be zero for cooling
    idx_x_heat = st.x > 0
    idx_x_cool = st.x <= 0
    c.qd_min_rn[idx_x_heat] = 0
    c.qd_max_rn[idx_x_cool] = 0
    c.qd_min_dc[idx_x_heat] = 0
    c.qd_max_dc[idx_x_cool] = 0
    # Where neither heating nor cooling discharging is possible
    idx_no_potential_rn = np.where(np.logical_or(np.logical_and(idx_x_heat, c.qd_max_rn <= 0),
                                                 np.logical_and(idx_x_cool, c.qd_min_rn >= 0)))
    idx_no_potential_dc = np.where(np.logical_or(np.logical_and(idx_x_heat, c.qd_max_dc <= 0),
                                                 np.logical_and(idx_x_cool, c.qd_min_dc >= 0)))
    c.qd_max_rn[idx_no_potential_rn] = 0
    c.qd_min_rn[idx_no_potential_rn] = 0
    c.qd_max_dc[idx_no_potential_dc] = 0
    c.qd_min_dc[idx_no_potential_dc] = 0


def mention_viols(st: State, c: Control, s: Scenario, disch: bool = True, duty_cycle: bool = True, prec: int = 4,
                  tol: float = 0):
    """Mention violations of discharging thermal power (>= 0 or <= 0) and simultaneous charging/discharging behavior,
    prec represents the decimal precision used when making these assessments and tol is the maximum duty cycle violation
     accepted."""

    # Setup
    n = len(s.t)

    if disch:
        # Mention violations of discharging criteria
        idx_x_heat = st.x >= 0
        idx_x_cool = st.x < 0
        disch_voil_rn = np.zeros(n)
        disch_voil_dc = np.zeros(n)
        disch_voil_rn[idx_x_heat] = np.round(-cvx.neg(c.qd_rn[idx_x_heat]).value, prec)
        disch_voil_rn[idx_x_cool] = np.round(cvx.pos(c.qd_rn[idx_x_cool]).value, prec)
        disch_voil_dc[idx_x_heat] = np.round(-cvx.neg(c.qd_dc[idx_x_heat]).value, prec)
        disch_voil_dc[idx_x_cool] = np.round(cvx.pos(c.qd_dc[idx_x_cool]).value, prec)
        if disch_voil_rn.any():
            warn(f"The solution to the CVX optimization problem violates the tank discharging power "
                 f"criteria for the ReNEWW House at {len(disch_voil_rn[disch_voil_rn != 0])}/{n} points with a "
                 f"maximum violation of {abs_max(disch_voil_rn)} kW.")
        if disch_voil_dc.any():
            warn(f"The solution to the CVX optimization problem violates the tank discharging power "
                 f"criteria for the DC House at {len(disch_voil_dc[disch_voil_dc != 0])}/{n} points with a "
                 f"maximum violation of {abs_max(disch_voil_dc)} kW.")

    if duty_cycle:
        # Mention violations of duty cycle criteria
        simul_charge_disch_voil = c.dc + c.dd_rn + c.dd_dc - 1
        simul_charge_disch_voil_max = round(max(simul_charge_disch_voil), prec)
        if simul_charge_disch_voil_max > tol:
            warn(f"The solution to the CVX optimization problem violates the simultaneous tank "
                 f"charging and discharging duty cycle criteria at "
                 f"{len(simul_charge_disch_voil[simul_charge_disch_voil > tol])}/{n} points with a maximum violation of "
                 f"{simul_charge_disch_voil_max}.")

        simul_charge_gshp_voil = c.dc + c.d_gshp_rn + c.d_gshp_dc - 1
        simul_charge_gshp_voil_max = round(max(simul_charge_gshp_voil), prec)
        if simul_charge_gshp_voil_max > 0:
            warn(f"The solution to the CVX optimization problem violates the simultaneous GSHP tank "
                 f"charging and delivery to ReNEWW/DC duty cycle criteria at "
                 f"{len(simul_charge_gshp_voil[simul_charge_gshp_voil > 0])}/{n} points with a maximum violation of "
                 f"{simul_charge_gshp_voil_max}.")


def solve_cvx_scp(s: Scenario, tk: Tank, x_0: float, cvx_state_fallback: State | None = None,
              cvx_control_fallback: Control | None = None) -> tuple[State, Control]:
    """Solves the optimization problem given the scenario, tank model, and initial state. In this optimization
    GSHP to DC house pathway is not allowed, so non-convex duty cycle constraints preventing simultaneous charging and
    discharging are enforced through a sequential convex programming loop, optimality is not guaranteed and the
    optimization may fall back on the solution to the relaxed problem if the SCP loop fails.

    Parameters:
        s: An instance of the Scenario class for the optimization horizon which has been reduced with Scenario.reduce()
        tk : An instance of the Tank class used in the optimization
        x_0: The initial tank energy in kWh
        cvx_state_fallback: Instance of the State class used as a fallback if the main optimization or SCP loop fails
        cvx_control_fallback: Instance of the State class used as a fallback if the main optimization or SCP loop fails

    Outputs:
        cvx_state: An instance of the State class with the recorded tank state
        cvx_control: An instance of the Control class with the recorded control inputs
    """
    # < Setup >
    # Number of timesteps in optimization horizon
    n = len(s.t)
    # Heating load indices for ReNEWW House
    h_rn = np.where(s.rn.load > 0)[0]
    # Cooling load indices for ReNEWW House
    c_rn = np.where(s.rn.load <= 0)[0]
    # Zero load indices for ReNEWW House
    z_rn = np.where(s.rn.load == 0)[0]
    # Heating load indices for DC House
    h_dc = np.where(s.dc.load > 0)[0]
    # Cooling load indices for DC House
    c_dc = np.where(s.dc.load <= 0)[0]
    # Zero load indices for DC House
    z_dc = np.where(s.dc.load == 0)[0]

    # < CVX variables >
    # Tank state
    x = cvx.Variable(n + 1, name="x")

    # Electrical power
    pc = cvx.Variable(n, name="pc", nonneg=True)
    p_gshp_rn = cvx.Variable(n, name="p_gshp_rn", nonneg=True)
    p_ashp = cvx.Variable(n, name="p_ashp", nonneg=True)
    p_aux_gshp = cvx.Variable(n, name="p_aux_gshp", nonneg=True)
    p_aux_ashp = cvx.Variable(n, name="p_aux_ashp", nonneg=True)

    # Thermal power
    qc = cvx.Variable(n, name="qc")
    qd_rn = cvx.Variable(n, name="qd_rn ")
    qd_dc = cvx.Variable(n, name="qd_dc")
    slack_qd_rn = cvx.Variable(n, name="slack_qd_rn", nonneg=True)
    slack_qd_dc = cvx.Variable(n, name="slack_qd_dc", nonneg=True)
    q_gshp_rn = cvx.Variable(n, name="q_gshp_rn")
    q_ashp = cvx.Variable(n, name="q_ashp")

    # Efficiency
    eta_d_rn = cvx.Variable(n, name="eta_d_rn")
    eta_d_dc = cvx.Variable(n, name="eta_d_dc")

    # Duty cycle
    dc = cvx.Variable(n, name="dc", nonneg=True)
    d_gshp_rn = cvx.Variable(n, name="d_gshp_rn", nonneg=True)
    d_ashp = cvx.Variable(n, name="d_ashp", nonneg=True)
    d_aux_gshp = cvx.Variable(n, name="d_aux_gshp", nonneg=True)
    d_aux_ashp = cvx.Variable(n, name="d_aux_ashp", nonneg=True)

    # < CVX Expressions >
    # Discharging power limits
    qd_max_rn = tk.qd_max_rn[0]*x[:-1] + tk.qd_max_rn[1]
    qd_min_rn = tk.qd_min_rn[0]*x[:-1] + tk.qd_min_rn[1]
    qd_max_dc = tk.qd_max_dc[0]*x[:-1] + tk.qd_max_dc[1]
    qd_min_dc = tk.qd_min_dc[0]*x[:-1] + tk.qd_min_dc[1]

    # Total electrical power (discounting pumping power of discharging)
    p_tot_rn = pc + p_gshp_rn + p_aux_gshp
    p_tot_dc = p_ashp + p_aux_ashp

    # Electricity cost expression for the objective function
    elec_cost = calc_elec_cost(s, p_tot_rn, p_tot_dc)

    # Terminal price of battery energy for the objective function
    pi_e_rn_n = s.rn.rate_plan.energy_rate[-1]
    
    # < Objective Function >
    # Desired ending tank state so final and initial state in simulation are equal
    if s.x_0 is None:
        # If initial state is not given for omniscient optimal scenario
        x_f_des = x[0]
    else:
        # If an initial state was set or chosen by omniscient
        x_f_des = s.x_0
    # Cost for energy lost in tank from initial state
    if s.term_weight is not None:
        if s.rn.load[-1] >= 0:
            # In heating mode, reward x[n] > x[0]
            term_cost = s.term_weight*pi_e_rn_n*cvx.pos(x_f_des - x[-2])
        else:
            # In cooling mode, reward x[n] < x[0]
            term_cost = s.term_weight*pi_e_rn_n*cvx.pos(x[-2] - x_f_des)
    else:
        term_cost = 0

    # Comfort cost
    comf_cost_rn = s.comfort_weight*cvx.sum(cvx.abs(q_gshp_rn + p_aux_gshp + qd_rn - s.rn.load))
    comf_cost_dc = s.comfort_weight*cvx.sum(cvx.abs(q_ashp + p_aux_ashp + qd_dc - s.dc.load))

    # Discharging violation cost (through slack variables)
    disch_viol_cost = s.disch_viol_weight*(cvx.sum(slack_qd_rn[h_rn]/np.polyval(tk.qd_max_rn, tk.x_max)) +
                                           cvx.sum(slack_qd_rn[c_rn]/np.polyval(-tk.qd_min_rn, tk.x_min)) +
                                           cvx.sum(slack_qd_dc[h_dc]/np.polyval(tk.qd_max_dc, tk.x_max)) +
                                           cvx.sum(slack_qd_dc[c_dc]/np.polyval(-tk.qd_min_dc, tk.x_min)))

    # The objective function
    objective = cvx.Minimize(elec_cost + term_cost + comf_cost_rn + comf_cost_dc + disch_viol_cost)

    # < Tank Constraints >
    constraints = []
    # Initial tank state (let optimization decide initial tank state if it is not given)
    if not np.isnan(x_0):
        constraints += [x[0] == x_0]
    # Final tank state (if term_weight is not given)
    if s.term_weight is None:
        constraints += [x[-2] == x_f_des]
    # Dynamics of the tank
    constraints += [x[1:] == tk.a*x[:-1] + tk.b*(qc - (qd_rn + qd_dc))]
    # Energy storage limits of the tank
    constraints += [tk.x_min <= x, x <= tk.x_max]

    # < Heat pump constraints (heating mode) >
    # GSHP power limits
    constraints += [pc[h_rn] + p_gshp_rn[h_rn] <= s.rn.p_hp_max_heat[h_rn]]
    # GSHP thermal power
    constraints += [q_gshp_rn[h_rn] == cvx.multiply(p_gshp_rn[h_rn], s.rn.cop_heat[h_rn])]
    # Aux heat limits
    constraints += [p_aux_gshp[h_rn] <= s.rn.p_aux_max[h_rn]]

    # ASHP power limits
    constraints += [p_ashp[h_dc] <= s.dc.p_hp_max_heat[h_dc]]
    # ASHP thermal power
    constraints += [q_ashp[h_dc] == cvx.multiply(p_ashp[h_dc], s.dc.cop_heat[h_dc])]
    # Aux heat limits
    constraints += [p_aux_ashp[h_dc] <= s.dc.p_aux_max[h_dc]]

    # < Discharging constraints (heating mode) >
    # Discharging efficiency
    constraints += [eta_d_rn[h_rn]*tk.pd_max_rn == qd_max_rn[h_rn]]
    constraints += [eta_d_dc[h_dc]*tk.pd_max_dc == qd_max_dc[h_dc]]
    # Discharging thermal power, enforce qd <= qd_max with slack variables in case sign of qd_max goes negative
    constraints += [0 <= qd_rn[h_rn], qd_rn[h_rn] <= qd_max_rn[h_rn] + slack_qd_rn[h_rn]]
    constraints += [0 <= qd_dc[h_dc], qd_dc[h_dc] <= qd_max_dc[h_dc] + slack_qd_dc[h_dc]]

    # < Heat pump constraints (cooling mode) >
    # GSHP power limits
    constraints += [pc[c_rn] + p_gshp_rn[c_rn] <= s.rn.p_hp_max_cool[c_rn]]
    # GSHP thermal power
    constraints += [q_gshp_rn[c_rn] == cvx.multiply(p_gshp_rn[c_rn], s.rn.cop_cool[c_rn])]
    # Aux heat limits
    constraints += [p_aux_gshp[c_rn] == 0]

    # ASHP power limits
    constraints += [p_ashp[c_dc] <= s.dc.p_hp_max_cool[c_dc]]
    # ASHP thermal power
    constraints += [q_ashp[c_dc] == cvx.multiply(p_ashp[c_dc], s.dc.cop_cool[c_dc])]
    # Aux heat limits
    constraints += [p_aux_ashp[c_dc] == 0]

    # < Heat pump constraints (general) >
    # GSHP thermal capacity limits (bounded by -inf, inf if excluded from config)
    constraints += [qc + q_gshp_rn <= s.rn.q_hp_max_heat, s.rn.q_hp_max_cool <= qc + q_gshp_rn]

    # ASHP thermal capacity limits (bounded by -inf, inf if excluded from config)
    constraints += [q_ashp <= s.dc.q_hp_max_heat, s.dc.q_hp_max_cool <= q_ashp]

    # < Discharging constraints (cooling mode) >
    # Discharging efficiency
    constraints += [eta_d_rn[c_rn]*tk.pd_max_rn == qd_min_rn[c_rn]]
    constraints += [eta_d_dc[c_dc]*tk.pd_max_dc == qd_min_dc[c_dc]]
    # Discharging thermal power, enforce qd_min <= qd with slack variables in case sign of qd_min goes positive
    constraints += [qd_min_rn[c_rn] <= qd_rn[c_rn] + slack_qd_rn[c_rn], qd_rn[c_rn] <= 0]
    constraints += [qd_min_dc[c_dc] <= qd_dc[c_dc] + slack_qd_dc[c_dc], qd_dc[c_dc] <= 0]

    # < Heat pump constraints (zero-load) >
    # Force off to prevent opposite sign heat pump and discharging heat flows
    constraints += [p_gshp_rn[z_rn] == 0]
    constraints += [p_ashp[z_dc] == 0]

    # < Charging constraints >
    constraints += [pc >= 0]
    constraints += [cvx.multiply(s.rn.cop_cool, pc) <= qc, qc <= cvx.multiply(s.rn.cop_heat, pc)]

    # < Duty cycle constraints >
    # Duty cycle definitions (heating)
    constraints += [pc[h_rn] == cvx.multiply(dc[h_rn], s.rn.p_hp_max_heat[h_rn])]
    constraints += [p_gshp_rn[h_rn] == cvx.multiply(d_gshp_rn[h_rn], s.rn.p_hp_max_heat[h_rn])]
    constraints += [p_ashp[h_dc] == cvx.multiply(d_ashp[h_dc], s.dc.p_hp_max_heat[h_dc])]

    # Duty cycle definitions (cooling)
    constraints += [pc[c_rn] == cvx.multiply(dc[c_rn], s.rn.p_hp_max_cool[c_rn])]
    constraints += [p_gshp_rn[c_rn] == cvx.multiply(d_gshp_rn[c_rn], s.rn.p_hp_max_cool[c_rn])]
    constraints += [p_ashp[c_dc] == cvx.multiply(d_ashp[c_dc], s.dc.p_hp_max_cool[c_dc])]

    # Duty cycle definitions (general)
    constraints += [p_aux_gshp == cvx.multiply(d_aux_gshp, s.rn.p_aux_max)]
    constraints += [p_aux_ashp == cvx.multiply(d_aux_ashp, s.dc.p_aux_max)]
    if not (s.rn.p_aux_max != 0).any():
        constraints += [d_aux_gshp == 0]
    if not (s.dc.p_aux_max != 0).any():
        constraints += [d_aux_ashp == 0]
    constraints += [dc <= 1]
    constraints += [d_gshp_rn <= 1]
    constraints += [d_ashp <= 1]
    constraints += [d_aux_gshp <= 1]
    constraints += [d_aux_ashp <= 1]

    # GSHP cannot charge tanks and deliver heating/cooling to ReNEWW simultaneously
    constraints += [dc + d_gshp_rn <= 1]
    # constraints += [qd_rn[h_rn] + qd_dc[h_rn] <= qd_max_rn[h_rn]]
    # constraints += [qd_rn[c_rn] + qd_dc[c_rn] >= qd_min_rn[c_rn]]

    # < Energy balance constraints >
    if s.comfort_weight == 0:
        # Enforce hard energy balance equality
        constraints += [q_ashp + p_aux_ashp + qd_dc == s.dc.load]
        constraints += [q_gshp_rn + p_aux_gshp + qd_rn == s.rn.load]

    # < Solve problem >
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver=s.solver)
    # Return status and whether solution is optimal
    status, success, failure = (prob.status, prob.status == cvx.OPTIMAL, prob.status != cvx.OPTIMAL)

    # Parameters for SCP loop
    tol = 0.05
    iter = 100
    alpha = 0.9
    beta = 1.03
    if success:
        # < Save results as fallback >
        cvx_state_fallback = State(x.value[:-1])
        cvx_control_fallback = Control(pc.value, np.zeros(n), np.zeros(n), p_gshp_rn.value, np.zeros(n), p_ashp.value,
                              p_aux_gshp.value, p_aux_ashp.value, dc.value, np.zeros(n), np.zeros(n), d_gshp_rn.value,
                              np.zeros(n), d_ashp.value, d_aux_gshp.value, d_aux_ashp.value, qc.value, qd_rn.value,
                              qd_dc.value, q_gshp_rn.value, np.zeros(n), q_ashp.value, qd_max_rn.value,
                              qd_min_rn.value, qd_max_dc.value, qd_min_dc.value, eta_d_rn.value, eta_d_dc.value)
        # Calculate discharging duty cycle and power
        calc_duty_cycle_power(cvx_state_fallback, cvx_control_fallback, tk)

        # < Use results to define parameters for qd_max and qd_min >
        qd_max_rn_param = cvx.Parameter(n, value=qd_max_rn.value.copy(), name="qd_max_rn_param")
        qd_min_rn_param = cvx.Parameter(n, value=qd_min_rn.value.copy(), name="qd_min_rn_param")
        qd_max_dc_param = cvx.Parameter(n, value=qd_max_dc.value.copy(), name="qd_max_dc_param")
        qd_min_dc_param = cvx.Parameter(n, value=qd_min_dc.value.copy(), name="qd_min_dc_param")

        # < Add power and duty cycle parameters for discharging >
        pd_rn = cvx.Variable(n, name="pd_rn", nonneg=True)
        pd_dc = cvx.Variable(n, name="pd_dc", nonneg=True)
        dd_rn = cvx.Variable(n, name="dd_rn", nonneg=True)
        dd_dc = cvx.Variable(n, name="dd_dc", nonneg=True)

        # < Add discharging power to objective function >
        # Total power expressions
        p_tot_rn = pd_rn + pc + p_gshp_rn + p_aux_gshp
        p_tot_dc = pd_dc + p_ashp + p_aux_ashp
        # Electricity cost expression for the objective function
        elec_cost = calc_elec_cost(s, p_tot_rn, p_tot_dc)
        # The redefined objective function
        objective = cvx.Minimize(elec_cost + term_cost + comf_cost_rn + comf_cost_dc + disch_viol_cost)

        # < Added constraints >
        # Discharging electrical power >
        constraints += [pd_rn == cvx.multiply(dd_rn, tk.pd_max_rn)]
        constraints += [pd_dc == cvx.multiply(dd_dc, tk.pd_max_dc)]
        constraints += [pd_rn <= tk.pd_max_rn]
        constraints += [pd_dc <= tk.pd_max_dc]

        # Duty cycle
        constraints += [dd_rn <= 1]
        constraints += [dd_dc <= 1]
        constraints += [qd_rn[h_rn] == cvx.multiply(dd_rn[h_rn], qd_max_rn_param[h_rn])]
        constraints += [qd_rn[c_rn] == cvx.multiply(dd_rn[c_rn], qd_min_rn_param[c_rn])]
        constraints += [qd_dc[h_dc] == cvx.multiply(dd_dc[h_dc], qd_max_dc_param[h_dc])]
        constraints += [qd_dc[c_dc] == cvx.multiply(dd_dc[c_dc], qd_min_dc_param[c_dc])]

        # Cannot simultaneously charge and discharge tanks
        constraints += [dc + dd_rn + dd_dc <= 1]
        # Cannot simultaneously

        # < Redefine problem >
        prob = cvx.Problem(objective, constraints)

        within_tol = False
        max_resid_tot_prev = np.inf
        # Begin SCP loop
        for _ in range(iter):
            # Use previous results to inform new optimization
            prob.solve(solver=s.solver, warm_start=True)
            # Return status and whether solution is optimal
            status, success, failure = (prob.status, prob.status == cvx.OPTIMAL, prob.status != cvx.OPTIMAL)

            if success:
                # < Recompute qd_max, qd_min based on results >
                # Define new guess as real values
                qd_max_rn_new = qd_max_rn.value.copy()
                qd_min_rn_new = qd_min_rn.value.copy()
                qd_max_dc_new = qd_max_dc.value.copy()
                qd_min_dc_new = qd_min_dc.value.copy()

                # Calculate residual
                max_resid_tot = (
                    np.percentile(np.abs(np.concatenate((qd_max_rn_new[h_rn], qd_min_rn_new[c_rn],
                                                  qd_max_dc_new[h_dc], qd_min_dc_new[c_dc])) -
                                  np.concatenate((qd_max_rn_param.value[h_rn], qd_min_rn_param.value[c_rn],
                                                  qd_max_dc_param.value[h_dc], qd_min_dc_param.value[c_dc]))), 97))

                # Correct discharging duty cycle and electrical power
                idx_x_heat = x.value[:-1] >= 0
                idx_x_cool = x.value[:-1] < 0
                dd_rn.value[idx_x_heat] = safe_div(qd_rn.value[idx_x_heat], qd_max_rn.value[idx_x_heat], 0)
                dd_rn.value[idx_x_cool] = safe_div(qd_rn.value[idx_x_cool], qd_min_rn.value[idx_x_cool], 0)
                dd_dc.value[idx_x_heat] = safe_div(qd_dc.value[idx_x_heat], qd_max_dc.value[idx_x_heat], 0)
                dd_dc.value[idx_x_cool] = safe_div(qd_dc.value[idx_x_cool], qd_min_dc.value[idx_x_cool], 0)
                pd_rn.value = dd_rn.value*tk.pd_max_rn
                pd_dc.value = dd_dc.value*tk.pd_max_dc

                # Calculate violations of duty cycle criteria
                simul_charge_disch_voil = dc.value + dd_rn.value + dd_dc.value - 1
                simul_charge_disch_voil_max = round(max(simul_charge_disch_voil), 4)

                if simul_charge_disch_voil_max < tol:
                    # Break if tolerance and duty cycle criteria is satisfied
                    within_tol = True
                    break
                if max_resid_tot > max_resid_tot_prev:
                    alpha = alpha/beta
                max_resid_tot_prev = max_resid_tot

                # Assign new values to parameters
                qd_max_rn_param.value = alpha*qd_max_rn_new + (1 - alpha)*qd_max_rn_param.value
                qd_min_rn_param.value = alpha*qd_min_rn_new + (1 - alpha)*qd_min_rn_param.value
                qd_max_dc_param.value = alpha*qd_max_dc_new + (1 - alpha)*qd_max_dc_param.value
                qd_min_dc_param.value = alpha*qd_min_dc_new + (1 - alpha)*qd_min_dc_param.value

            else:
                break

    # < Check status >
    if failure:
        if (cvx_state_fallback is not None) and (cvx_control_fallback is not None):
            warn(f"The solution to the CVX optimization problem within the SCP loop is infeasible, status is "
                 f"'{status}'.\nThe optimization has fallen back to the initial (pre-loop) solution.")
            cvx_state = cvx_state_fallback
            cvx_control = cvx_control_fallback
        else:
            warn(f"The solution to the CVX optimization problem is infeasible, status is "
                 f"'{status}'.\nBaseline (no TES) operation has been assumed and state has been propagated "
                 f"for this case.")
            # Preserve initial state
            cvx_state = State.initial(s)
            # Set control values using no TES baseline logic
            cvx_control = simulate_no_tes_baseline(s)[1]
    else:
        # Check tolerance
        if not within_tol:
            warn(f"The solution to the CVX optimization problem did not meet the specified tolerance "
                 f"within {iter} iterations, the maximum residual is {max_resid_tot} kW with alpha = {alpha}.")
        # Instantiate state with optimal values
        cvx_state = State(x.value[:-1])
        
        # Instantiate control with values
        cvx_control = Control(pc.value, pd_rn.value, pd_dc.value, p_gshp_rn.value, np.zeros(n), p_ashp.value,
                              p_aux_gshp.value, p_aux_ashp.value, dc.value, dd_rn.value, dd_dc.value, d_gshp_rn.value,
                              np.zeros(n), d_ashp.value, d_aux_gshp.value, d_aux_ashp.value, qc.value, qd_rn.value,
                              qd_dc.value, q_gshp_rn.value, np.zeros(n), q_ashp.value, qd_max_rn.value,
                              qd_min_rn.value, qd_max_dc.value, qd_min_dc.value, eta_d_rn.value, eta_d_dc.value)

    if failure:
        # Propagate initial state through with no control inputs
        for i in range(n):
            cvx_state.update(tk, cvx_control)

    # < Correct qd_max and qd_min >
    calc_disch_limits(cvx_state, cvx_control)

    # Mention violations of discharging and duty cycle criteria
    mention_viols(cvx_state, cvx_control, s, tol=tol)

    # Clamp discharging duty cycle and power
    cvx_control.dd_rn = clamp(cvx_control.dd_rn, 1.0)
    cvx_control.dd_dc = clamp(cvx_control.dd_dc, 1.0)

    return cvx_state, cvx_control


def solve_cvx_simplified(s: Scenario, tk: Tank, x_0: float) -> tuple[State, Control]:
    """Solves a convex optimization problem given the scenario, tank model, and initial state. In this formulation GSHP
    to DC is permitted and backcalculated from simultaneous charging/discharging behavior which emulates direct GSHP to
    DC operation.

    Parameters:
        s: An instance of the Scenario class for the optimization horizon which has been reduced with Scenario.reduce()
        tk : An instance of the Tank class used in the optimization
        x_0: The initial tank energy in kWh

    Outputs:
        cvx_state: An instance of the State class with the recorded tank state
        cvx_control: An instance of the Control class with the recorded control inputs
    """
    # < Setup >
    # Number of timesteps in optimization horizon
    n = len(s.t)
    # Heating load indices for ReNEWW House
    h_rn = np.where(s.rn.load > 0)[0]
    # Cooling load indices for ReNEWW House
    c_rn = np.where(s.rn.load <= 0)[0]
    # Zero load indices for ReNEWW House
    z_rn = np.where(s.rn.load == 0)[0]
    # Heating load indices for DC House
    h_dc = np.where(s.dc.load > 0)[0]
    # Cooling load indices for DC House
    c_dc = np.where(s.dc.load <= 0)[0]
    # Zero load indices for DC House
    z_dc = np.where(s.dc.load == 0)[0]
    # Heating load indices applying to both houses
    h = np.intersect1d(h_rn, h_dc)
    # h = np.union1d(h_rn, h_dc)
    # Cooling load indices applying to both houses
    c = np.intersect1d(c_rn, c_dc)
    # c = np.union1d(c_rn, c_dc)

    # < CVX variables >
    # Tank state
    x = cvx.Variable(n + 1, name="x")

    # Electrical power
    pc = cvx.Variable(n, name="pc", nonneg=True)
    p_gshp_rn = cvx.Variable(n, name="p_gshp_rn", nonneg=True)
    p_ashp = cvx.Variable(n, name="p_ashp", nonneg=True)
    p_aux_gshp = cvx.Variable(n, name="p_aux_gshp", nonneg=True)
    p_aux_ashp = cvx.Variable(n, name="p_aux_ashp", nonneg=True)

    # Thermal power
    qc = cvx.Variable(n, name="qc")
    qd_rn = cvx.Variable(n, name="qd_rn ")
    qd_dc = cvx.Variable(n, name="qd_dc")
    slack_qd_rn = cvx.Variable(n, name="slack_qd_rn", nonneg=True)
    slack_qd_dc = cvx.Variable(n, name="slack_qd_dc", nonneg=True)
    q_gshp_rn = cvx.Variable(n, name="q_gshp_rn")
    q_ashp = cvx.Variable(n, name="q_ashp")

    # Efficiency
    eta_d_rn = cvx.Variable(n, name="eta_d_rn")
    eta_d_dc = cvx.Variable(n, name="eta_d_dc")

    # Duty cycle
    dc = cvx.Variable(n, name="dc", nonneg=True)
    d_gshp_rn = cvx.Variable(n, name="d_gshp_rn", nonneg=True)
    d_ashp = cvx.Variable(n, name="d_ashp", nonneg=True)
    d_aux_gshp = cvx.Variable(n, name="d_aux_gshp", nonneg=True)
    d_aux_ashp = cvx.Variable(n, name="d_aux_ashp", nonneg=True)

    # < CVX Expressions >
    # Discharging power limits
    qd_max_rn = tk.qd_max_rn[0]*x[:-1] + tk.qd_max_rn[1]
    qd_min_rn = tk.qd_min_rn[0]*x[:-1] + tk.qd_min_rn[1]
    qd_max_dc = tk.qd_max_dc[0]*x[:-1] + tk.qd_max_dc[1]
    qd_min_dc = tk.qd_min_dc[0]*x[:-1] + tk.qd_min_dc[1]

    # Total electrical power
    # Obtain a conservative nonnegative estimate of general discharging efficiency for both houses
    eta_d_rn_general = max(np.polyval(tk.qd_max_rn, tk.x_max), np.abs(np.polyval(tk.qd_max_rn, tk.x_min)))*s.soc_eta_guess
    eta_d_dc_general = max(np.polyval(tk.qd_max_dc, tk.x_max), np.abs(np.polyval(tk.qd_max_dc, tk.x_min)))*s.soc_eta_guess
    # Define discharging powers using this value
    pd_rn = cvx.abs(qd_rn)/eta_d_rn_general
    pd_dc = cvx.abs(qd_dc)/eta_d_dc_general
    p_tot_rn = pd_rn + pc + p_gshp_rn + p_aux_gshp
    p_tot_dc = pd_dc + p_ashp + p_aux_ashp

    # Electricity cost expression for the objective function
    elec_cost = calc_elec_cost(s, p_tot_rn, p_tot_dc)

    # < Objective Function >
    # Desired ending tank state so final and initial state in simulation are equal
    if s.x_0 is None:
        # If initial state is not given for omniscient optimal scenario
        x_f_des = x[0]
    else:
        # If an initial state was set or chosen by omniscient
        x_f_des = s.x_0
    # Cost for energy lost in tank from initial state
    if ((n < s.k_horiz) or (s.controllers == "Omniscient Optimal")) and (s.term_weight is not None):
        # Horizon is receding or full simulation is provided, final timestep is end of simulation
        term_cost = s.term_weight*(x_f_des - x[-2])**2
    elif s.depl_weight is not None:
        # Combined thermal load at end of horizon
        tot_load_n = s.rn.load[-1] + s.dc.load[-1]
        if s.depl_weight == 'auto':
            # Value tank depletion at price of thermal energy based on GSHP COP
            if tot_load_n > 0:
                depl_weight = 1/s.rn.cop_heat[-1]
            else:
                depl_weight = np.abs(1/s.rn.cop_cool[-1])
        else:
            depl_weight = s.depl_weight
        # Mean energy price for ReNEWW house over horizon
        pi_e_rn_avg = np.mean(s.rn.rate_plan.energy_rate)
        # Penalize tank depletion, but don't reward excess tank energy within horizon
        if tot_load_n > 0:
            # In overall heating mode, reward x[n] > x[0]
            term_cost = depl_weight*pi_e_rn_avg*cvx.pos(x_f_des - x[-2])
        else:
            # In overall cooling mode, reward x[n] < x[0]
            term_cost = depl_weight*pi_e_rn_avg*cvx.pos(x[-2] - x_f_des)
    else:
        term_cost = 0

    # Comfort cost
    comf_cost_rn = s.comfort_weight*cvx.sum(cvx.abs(q_gshp_rn + p_aux_gshp + qd_rn - s.rn.load))
    comf_cost_dc = s.comfort_weight*cvx.sum(cvx.abs(q_ashp + p_aux_ashp + qd_dc - s.dc.load))

    # Run the no TES baseline to estimate magnitude of electricity costs
    baseline_control = simulate_no_tes_baseline(s)[1]
    elec_cost_baseline = calc_elec_cost(s, c=baseline_control)['elec_cost_obj']
    elec_cost_baseline_hourly = elec_cost_baseline/(n*s.dt.total_seconds()/3600)

    # Discharging violation cost (through slack variables), scale by estimated electricity cost so weight is valid
    # for many different scenarios
    disch_viol_cost = s.disch_viol_weight*elec_cost_baseline_hourly*cvx.sum(cvx.maximum(slack_qd_rn, slack_qd_dc))

    # The objective function
    objective = cvx.Minimize(elec_cost + term_cost + comf_cost_rn + comf_cost_dc + disch_viol_cost)

    # < Tank Constraints >
    constraints = []
    # Initial tank state (let optimization decide initial tank state if it is not given)
    if not np.isnan(x_0):
        constraints += [x[0] == x_0]
    # Final tank state (if battery depletion/terminal state weights are not given)
    if (s.term_weight is None) and (s.depl_weight is None):
        constraints += [x[-2] == x_f_des]
    # Dynamics of the tank
    constraints += [x[1:] == tk.a*x[:-1] + tk.b*(qc - (qd_rn + qd_dc))]
    # Energy storage limits of the tank
    constraints += [tk.x_min <= x, x <= tk.x_max]

    # < Heat pump constraints (heating mode) >
    # GSHP power limits
    # constraints += [pc[h_rn] + p_gshp_rn[h_rn] <= s.rn.p_hp_max_heat[h_rn]]
    # GSHP thermal power
    constraints += [q_gshp_rn[h_rn] == cvx.multiply(p_gshp_rn[h_rn], s.rn.cop_heat[h_rn])]
    # Aux heat limits
    constraints += [p_aux_gshp[h_rn] <= s.rn.p_aux_max[h_rn]]

    # ASHP power limits
    constraints += [p_ashp[h_dc] <= s.dc.p_hp_max_heat[h_dc]]
    # ASHP thermal power
    constraints += [q_ashp[h_dc] == cvx.multiply(p_ashp[h_dc], s.dc.cop_heat[h_dc])]
    # Aux heat limits
    constraints += [p_aux_ashp[h_dc] <= s.dc.p_aux_max[h_dc]]

    # < Discharging constraints (heating mode) >
    # Discharging efficiency
    constraints += [eta_d_rn[h_rn]*tk.pd_max_rn == qd_max_rn[h_rn]]
    constraints += [eta_d_dc[h_dc]*tk.pd_max_dc == qd_max_dc[h_dc]]
    # Discharging thermal power, enforce qd <= qd_max with slack variables in case sign of qd_max goes negative
    constraints += [0 <= qd_rn[h_rn], qd_rn[h_rn] <= qd_max_rn[h_rn] + slack_qd_rn[h_rn]]
    constraints += [0 <= qd_dc[h_dc], qd_dc[h_dc] <= qd_max_dc[h_dc] + slack_qd_dc[h_dc]]
    # constraints += [qd_rn[h_rn] <= slack_qd_rn[h_rn], slack_qd_rn[h_rn] >= qd_max_rn[h_rn]]
    # constraints += [qd_dc[h_dc] <= slack_qd_dc[h_dc], slack_qd_dc[h_dc] >= qd_max_dc[h_dc]]

    # < Heat pump constraints (cooling mode) >
    # GSHP power limits
    # constraints += [pc[c_rn] + p_gshp_rn[c_rn] <= s.rn.p_hp_max_cool[c_rn]]
    # GSHP thermal power
    constraints += [q_gshp_rn[c_rn] == cvx.multiply(p_gshp_rn[c_rn], s.rn.cop_cool[c_rn])]
    # Aux heat limits
    constraints += [p_aux_gshp[c_rn] == 0]

    # ASHP power limits
    constraints += [p_ashp[c_dc] <= s.dc.p_hp_max_cool[c_dc]]
    # ASHP thermal power
    constraints += [q_ashp[c_dc] == cvx.multiply(p_ashp[c_dc], s.dc.cop_cool[c_dc])]
    # Aux heat limits
    constraints += [p_aux_ashp[c_dc] == 0]

    # < Heat pump constraints (general) >
    # GSHP thermal capacity limits (bounded by -inf, inf if excluded from config)
    constraints += [qc + q_gshp_rn <= s.rn.q_hp_max_heat, s.rn.q_hp_max_cool <= qc + q_gshp_rn]

    # ASHP thermal capacity limits (bounded by -inf, inf if excluded from config)
    constraints += [q_ashp <= s.dc.q_hp_max_heat, s.dc.q_hp_max_cool <= q_ashp]

    # < Discharging constraints (cooling mode) >
    # Discharging efficiency
    constraints += [eta_d_rn[c_rn]*tk.pd_max_rn == qd_min_rn[c_rn]]
    constraints += [eta_d_dc[c_dc]*tk.pd_max_dc == qd_min_dc[c_dc]]
    # Discharging thermal power, enforce qd_min <= qd with slack variables in case sign of qd_min goes positive
    constraints += [qd_min_rn[c_rn] <= qd_rn[c_rn] + slack_qd_rn[c_rn], qd_rn[c_rn] <= 0]
    constraints += [qd_min_dc[c_dc] <= qd_dc[c_dc] + slack_qd_dc[c_dc], qd_dc[c_dc] <= 0]
    # constraints += [qd_min_rn[c_rn] <= -slack_qd_rn[c_rn], -slack_qd_rn[c_rn] >= qd_rn[h_rn]]
    # constraints += [qd_min_dc[c_dc] <= -slack_qd_dc[c_dc], -slack_qd_dc[c_dc] >= qd_dc[h_dc]]

    # < Heat pump constraints (zero-load) >
    # Force off to prevent opposite sign heat pump and discharging heat flows
    constraints += [p_gshp_rn[z_rn] == 0]
    constraints += [p_ashp[z_dc] == 0]
    # constraints += [qd_rn[z_rn] == 0]
    # constraints += [qd_dc[z_dc] == 0]

    # < Charging constraints >
    # constraints += [cvx.multiply(s.rn.cop_cool, pc) <= qc, qc <= cvx.multiply(s.rn.cop_heat, pc)]
    constraints += [0 <= qc[h_rn], qc[h_rn] <= cvx.multiply(s.rn.cop_heat[h_rn], pc[h_rn])]
    constraints += [cvx.multiply(s.rn.cop_cool[c_rn], pc[c_rn]) <= qc[c_rn], qc[c_rn] <= 0]

    # < Duty cycle constraints >
    # Duty cycle definitions (heating)
    # constraints += [pc[h_rn] == cvx.multiply(dc[h_rn], s.rn.p_hp_max_heat[h_rn])]
    constraints += [p_gshp_rn[h_rn] == cvx.multiply(d_gshp_rn[h_rn], s.rn.p_hp_max_heat[h_rn])]
    constraints += [p_ashp[h_dc] == cvx.multiply(d_ashp[h_dc], s.dc.p_hp_max_heat[h_dc])]

    # Duty cycle definitions (cooling)
    # constraints += [pc[c_rn] == cvx.multiply(dc[c_rn], s.rn.p_hp_max_cool[c_rn])]
    constraints += [p_gshp_rn[c_rn] == cvx.multiply(d_gshp_rn[c_rn], s.rn.p_hp_max_cool[c_rn])]
    constraints += [p_ashp[c_dc] == cvx.multiply(d_ashp[c_dc], s.dc.p_hp_max_cool[c_dc])]

    # Duty cycle definitions (general)
    constraints += [dc >= cvx.maximum(qc/s.rn.p_hp_max_heat/s.rn.cop_heat, qc/s.rn.p_hp_max_cool/s.rn.cop_cool)]
    constraints += [pc >= cvx.maximum(qc/s.rn.cop_heat, qc/s.rn.cop_cool)]

    # Duty cycle definitions (general)
    constraints += [p_aux_gshp == cvx.multiply(d_aux_gshp, s.rn.p_aux_max)]
    constraints += [p_aux_ashp == cvx.multiply(d_aux_ashp, s.dc.p_aux_max)]
    if not (s.rn.p_aux_max != 0).any():
        constraints += [d_aux_gshp == 0]
    if not (s.dc.p_aux_max != 0).any():
        constraints += [d_aux_ashp == 0]
    constraints += [dc <= 1]
    constraints += [d_gshp_rn <= 1]
    constraints += [d_ashp <= 1]
    constraints += [d_aux_gshp <= 1]
    constraints += [d_aux_ashp <= 1]

    # GSHP cannot charge tanks and deliver heating/cooling to ReNEWW simultaneously
    constraints += [dc + d_gshp_rn <= 1]

    # Cannot simultaneously discharge to DC and ReNEWW houses
    constraints += [qd_rn[h] + qd_dc[h] <= qd_max_rn[h] + slack_qd_rn[h] + slack_qd_dc[h]]
    constraints += [qd_rn[c] + qd_dc[c] + slack_qd_rn[c] + slack_qd_dc[c] >= qd_min_rn[c]]
    # constraints += [qd_rn[h] + qd_dc[h] <= slack_qd_rn[h] + slack_qd_dc[h],
    #                 slack_qd_rn[h] + slack_qd_dc[h] >= qd_max_rn[h]]
    # constraints += [qd_min_rn[c] <= -slack_qd_rn[c] - slack_qd_dc[c],
    #                 -slack_qd_rn[c] - slack_qd_dc[c] >= qd_rn[c] + qd_dc[c]]

    # < Energy balance constraints >
    if s.comfort_weight == 0:
        # Enforce hard energy balance equality
        constraints += [q_ashp + p_aux_ashp + qd_dc == s.dc.load]
        constraints += [q_gshp_rn + p_aux_gshp + qd_rn == s.rn.load]

    # < Solve problem >
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver=s.solver)
    # Return status and whether solution is optimal
    status, success, failure = (prob.status, prob.status == cvx.OPTIMAL, prob.status != cvx.OPTIMAL)

    # < Check status >
    if failure or (elec_cost_baseline_hourly < s.elec_cost_thres):
        if failure:
            warn(f"The solution to the CVX optimization problem is infeasible, status is "
                 f"'{status}'.\nBaseline (no TES) operation has been assumed and state has been propagated "
                 f"for this case.")
        else:
            print(f"The baseline hourly cost for this optimization is ${round(elec_cost_baseline_hourly, 2)}, "
                  f"this is below the threshold \"elec_cost_thres\" of ${s.elec_cost_thres}, so baseline no TES "
                  f"operation has been assumed.")
        # Preserve initial state
        cvx_state = State.initial(s)
        # Set control values using no TES baseline logic
        cvx_control = simulate_no_tes_baseline(s)[1]
        # Propagate initial state through with no control inputs
        for i in range(n):
            cvx_state.update(tk, cvx_control)
    else:
        # Instantiate state with optimal values
        cvx_state = State(x.value[:-1])
        # Instantiate control with optimal values
        cvx_control = Control(pc.value, np.zeros(n), np.zeros(n), p_gshp_rn.value, np.zeros(n), p_ashp.value,
                              p_aux_gshp.value, p_aux_ashp.value, dc.value, np.zeros(n), np.zeros(n), d_gshp_rn.value,
                              np.zeros(n), d_ashp.value, d_aux_gshp.value, d_aux_ashp.value, qc.value, qd_rn.value,
                              qd_dc.value, q_gshp_rn.value, np.zeros(n), q_ashp.value, qd_max_rn.value,
                              qd_min_rn.value, qd_max_dc.value, qd_min_dc.value, eta_d_rn.value, eta_d_dc.value)

    # < Correct qd_max and qd_min >
    calc_disch_limits(cvx_state, cvx_control)

    # < Calculate charging and discharging duty cycle and power >
    calc_duty_cycle_power(cvx_state, cvx_control, tk, s)

    # < Calculate GSHP to DC quantities
    # Any excess of the duty cycle requirement entails GHSP to ReNEWW direct operation
    idx_viol = np.where(np.round(cvx_control.dc + cvx_control.dd_dc + cvx_control.dd_rn, 4) > 1)
    # Violations in heating
    h_rn_viol = np.intersect1d(h_rn, idx_viol)
    # Violations in cooling
    c_rn_viol = np.intersect1d(c_rn, idx_viol)
    # GSHP to DC heat transfer is governed by the lesser of qd_dc and qc
    cvx_control.q_gshp_dc[h_rn_viol] = np.minimum(cvx_control.qd_dc[h_rn_viol], cvx_control.qc[h_rn_viol])
    cvx_control.q_gshp_dc[c_rn_viol] = np.maximum(cvx_control.qd_dc[c_rn_viol], cvx_control.qc[c_rn_viol])
    cvx_control.p_gshp_dc[h_rn_viol] = np.divide(cvx_control.q_gshp_dc[h_rn_viol], s.rn.cop_heat[h_rn_viol])
    cvx_control.p_gshp_dc[c_rn_viol] = np.divide(cvx_control.q_gshp_dc[c_rn_viol], s.rn.cop_cool[c_rn_viol])
    cvx_control.d_gshp_dc[h_rn_viol] = np.divide(cvx_control.p_gshp_dc[h_rn_viol], s.rn.p_hp_max_heat[h_rn_viol])
    cvx_control.d_gshp_dc[c_rn_viol] = np.divide(cvx_control.p_gshp_dc[c_rn_viol], s.rn.p_hp_max_cool[c_rn_viol])

    # Subtract GSHP to DC heat transfer from qd_dc, qc to maintain energy balance, one quantity will go to zero
    cvx_control.qd_dc -= cvx_control.q_gshp_dc
    cvx_control.qc -= cvx_control.q_gshp_dc

    # Recalculate charging behavior
    cvx_control.pc[h_rn_viol] = np.divide(cvx_control.qc[h_rn_viol], s.rn.cop_heat[h_rn_viol])
    cvx_control.pc[c_rn_viol] = np.divide(cvx_control.qc[c_rn_viol], s.rn.cop_cool[c_rn_viol])
    cvx_control.dc[h_rn_viol] = np.divide(cvx_control.pc[h_rn_viol], s.rn.p_hp_max_heat[h_rn_viol])
    cvx_control.dc[c_rn_viol] = np.divide(cvx_control.pc[c_rn_viol], s.rn.p_hp_max_cool[c_rn_viol])
    if (np.round(cvx_control.pc, 4) < 0).any():
        warn(f"The GSHP to ReNEWW simultaneous charging and discharging constraint led to an infeasible charging "
             f"power setpoint at {len(cvx_control.pc[np.round(cvx_control.pc, 4) < 0])} points with a minimum of "
             f"{min(cvx_control.pc)}")

    # Recalculate discharging behavior
    calc_duty_cycle_power(cvx_state, cvx_control, tk)
    if (np.round(cvx_control.pd_dc, 4) < 0).any():
        warn(f"The GSHP to ReNEWW simultaneous charging and discharging constraint led to an infeasible discharging "
             f"power setpoint at {len(cvx_control.pc[np.round(cvx_control.pd_dc, 4) < 0])} points with a minimum of "
             f"{min(cvx_control.pd_dc)}")

    # Mention violations of discharging and duty cycle criteria
    mention_viols(cvx_state, cvx_control, s)

    # Clamp discharging duty cycle and power
    cvx_control.dd_rn = clamp(cvx_control.dd_rn, 1.0)
    cvx_control.dd_dc = clamp(cvx_control.dd_dc, 1.0)

    return cvx_state, cvx_control


def solve_cvx_mixedint(s: Scenario, tk: Tank, x_0: float) -> tuple[State, Control]:
    """Solves a convex optimization problem given the scenario, tank model, and initial state. In this formulation GSHP
    to DC is permitted and backcalculated from simultaneous charging/discharging behavior which emulates direct GSHP to
    DC operation, discharging power constraints are enforced exactly using the big M method of mixed-integer programming.

    Parameters:
        s: An instance of the Scenario class for the optimization horizon which has been reduced with Scenario.reduce()
        tk : An instance of the Tank class used in the optimization
        x_0: The initial tank energy in kWh

    Outputs:
        cvx_state: An instance of the State class with the recorded tank state
        cvx_control: An instance of the Control class with the recorded control inputs
    """
    # < Setup >
    # Number of timesteps in optimization horizon
    n = len(s.t)
    # Heating load indices for ReNEWW House
    h_rn = np.where(s.rn.load > 0)[0]
    # Cooling load indices for ReNEWW House
    c_rn = np.where(s.rn.load <= 0)[0]
    # Zero load indices for ReNEWW House
    z_rn = np.where(s.rn.load == 0)[0]
    # Heating load indices for DC House
    h_dc = np.where(s.dc.load > 0)[0]
    # Cooling load indices for DC House
    c_dc = np.where(s.dc.load <= 0)[0]
    # Zero load indices for DC House
    z_dc = np.where(s.dc.load == 0)[0]
    # Heating load indices applying to both houses
    h = np.intersect1d(h_rn, h_dc)
    # Cooling load indices applying to both houses
    c = np.intersect1d(c_rn, c_dc)

    # Calculate absolute max and min values for discharging for big M
    qd_max_rn_abs = max(np.abs([np.polyval(tk.qd_max_rn, tk.x_max), np.polyval(tk.qd_max_rn, tk.x_min)]))
    qd_max_dc_abs = max(np.abs([np.polyval(tk.qd_max_dc, tk.x_max), np.polyval(tk.qd_max_dc, tk.x_min)]))
    qd_min_rn_abs = max(np.abs([np.polyval(tk.qd_min_rn, tk.x_min), np.polyval(tk.qd_min_rn, tk.x_max)]))
    qd_min_dc_abs = max(np.abs([np.polyval(tk.qd_min_dc, tk.x_min), np.polyval(tk.qd_min_dc, tk.x_max)]))

    # < CVX variables >
    # Tank state
    x = cvx.Variable(n + 1, name="x")
    # Boolean variables for big M
    # For if qd_max_rn is positive
    m_qd_rn_heat = cvx.Variable(n, name="m_qd_rn_heat", boolean=True)
    m_qd_dc_heat = cvx.Variable(n, name="m_qd_dc_heat", boolean=True)
    m_qd_rn_cool = cvx.Variable(n, name="m_qd_rn_cool", boolean=True)
    m_qd_dc_cool = cvx.Variable(n, name="m_qd_dc_cool", boolean=True)

    # Electrical power
    pc = cvx.Variable(n, name="pc", nonneg=True)
    p_gshp_rn = cvx.Variable(n, name="p_gshp_rn", nonneg=True)
    p_ashp = cvx.Variable(n, name="p_ashp", nonneg=True)
    p_aux_gshp = cvx.Variable(n, name="p_aux_gshp", nonneg=True)
    p_aux_ashp = cvx.Variable(n, name="p_aux_ashp", nonneg=True)

    # Thermal power
    qc = cvx.Variable(n, name="qc")
    qd_rn = cvx.Variable(n, name="qd_rn ")
    qd_dc = cvx.Variable(n, name="qd_dc")
    q_gshp_rn = cvx.Variable(n, name="q_gshp_rn")
    q_ashp = cvx.Variable(n, name="q_ashp")
    # Epigraph variables for discharging constraints
    e_rn = cvx.Variable(n, name="e_rn", nonneg=True)
    e_dc = cvx.Variable(n, name="e_dc", nonneg=True)
    # e_qd_heat = cvx.Variable(n, name="e_qd_heat", nonneg=True)
    # e_qd_cool = cvx.Variable(n, name="e_qd_cool", nonneg=True)

    # Efficiency
    eta_d_rn = cvx.Variable(n, name="eta_d_rn")
    eta_d_dc = cvx.Variable(n, name="eta_d_dc")

    # Duty cycle
    dc = cvx.Variable(n, name="dc", nonneg=True)
    d_gshp_rn = cvx.Variable(n, name="d_gshp_rn", nonneg=True)
    d_ashp = cvx.Variable(n, name="d_ashp", nonneg=True)
    d_aux_gshp = cvx.Variable(n, name="d_aux_gshp", nonneg=True)
    d_aux_ashp = cvx.Variable(n, name="d_aux_ashp", nonneg=True)

    # < CVX Expressions >
    # Discharging power limits
    qd_max_rn = tk.qd_max_rn[0]*x[:-1] + tk.qd_max_rn[1]
    qd_min_rn = tk.qd_min_rn[0]*x[:-1] + tk.qd_min_rn[1]
    qd_max_dc = tk.qd_max_dc[0]*x[:-1] + tk.qd_max_dc[1]
    qd_min_dc = tk.qd_min_dc[0]*x[:-1] + tk.qd_min_dc[1]

    # Total electrical power
    # Obtain a conservative nonnegative estimate of general discharging efficiency for both houses
    eta_d_rn_general = max(np.polyval(tk.qd_max_rn, tk.x_max), np.abs(np.polyval(tk.qd_max_rn, tk.x_min)))*s.soc_eta_guess
    eta_d_dc_general = max(np.polyval(tk.qd_max_dc, tk.x_max), np.abs(np.polyval(tk.qd_max_dc, tk.x_min)))*s.soc_eta_guess
    # Define discharging powers using this value
    pd_rn = cvx.abs(qd_rn)/eta_d_rn_general
    pd_dc = cvx.abs(qd_dc)/eta_d_dc_general
    p_tot_rn = pd_rn + pc + p_gshp_rn + p_aux_gshp
    p_tot_dc = pd_dc + p_ashp + p_aux_ashp

    # Electricity cost expression for the objective function
    elec_cost = calc_elec_cost(s, p_tot_rn, p_tot_dc)

    # Terminal price of battery energy for the objective function
    pi_e_rn_n = s.rn.rate_plan.energy_rate[-1]

    # < Objective Function >
    # Desired ending tank state so final and initial state in simulation are equal
    if s.x_0 is None:
        # If initial state is not given for omniscient optimal scenario
        x_f_des = x[0]
    else:
        # If an initial state was set or chosen by omniscient
        x_f_des = s.x_0
    # Cost for energy lost in tank from initial state
    if s.term_weight is not None:
        if s.rn.load[-1] + s.dc.load[-1] >= 0:
            # In overall heating mode, reward x[n] > x[0]
            term_cost = s.term_weight*pi_e_rn_n*cvx.pos(x_f_des - x[-2])
        else:
            # In overall cooling mode, reward x[n] < x[0]
            term_cost = s.term_weight*pi_e_rn_n*cvx.pos(x[-2] - x_f_des)
    else:
        term_cost = 0

    # Comfort cost
    comf_cost_rn = s.comfort_weight*cvx.sum(cvx.abs(q_gshp_rn + p_aux_gshp + qd_rn - s.rn.load))
    comf_cost_dc = s.comfort_weight*cvx.sum(cvx.abs(q_ashp + p_aux_ashp + qd_dc - s.dc.load))

    # The objective function
    objective = cvx.Minimize(elec_cost + term_cost + comf_cost_rn + comf_cost_dc)

    # < Tank Constraints >
    constraints = []
    # Initial tank state (let optimization decide initial tank state if it is not given)
    if not np.isnan(x_0):
        constraints += [x[0] == x_0]
    # Final tank state (if term_weight is not given)
    if s.term_weight is None:
        constraints += [x[-2] == x_f_des]
    # Dynamics of the tank
    constraints += [x[1:] == tk.a*x[:-1] + tk.b*(qc - (qd_rn + qd_dc))]
    # Energy storage limits of the tank
    constraints += [tk.x_min <= x, x <= tk.x_max]

    # < Heat pump constraints (heating mode) >
    # GSHP power limits
    # constraints += [pc[h_rn] + p_gshp_rn[h_rn] <= s.rn.p_hp_max_heat[h_rn]]
    # GSHP thermal power
    constraints += [q_gshp_rn[h_rn] == cvx.multiply(p_gshp_rn[h_rn], s.rn.cop_heat[h_rn])]
    # Aux heat limits
    constraints += [p_aux_gshp[h_rn] <= s.rn.p_aux_max[h_rn]]

    # ASHP power limits
    constraints += [p_ashp[h_dc] <= s.dc.p_hp_max_heat[h_dc]]
    # ASHP thermal power
    constraints += [q_ashp[h_dc] == cvx.multiply(p_ashp[h_dc], s.dc.cop_heat[h_dc])]
    # Aux heat limits
    constraints += [p_aux_ashp[h_dc] <= s.dc.p_aux_max[h_dc]]

    # < Discharging constraints (heating mode) >
    # Discharging efficiency
    constraints += [eta_d_rn[h_rn]*tk.pd_max_rn == qd_max_rn[h_rn]]
    constraints += [eta_d_dc[h_dc]*tk.pd_max_dc == qd_max_dc[h_dc]]
    # Discharging thermal power, use the big M method to enforce exact bounds
    constraints += [
        qd_max_rn[h_rn] <= qd_max_rn_abs*m_qd_rn_heat[h_rn],
        qd_max_rn[h_rn] >= -qd_max_rn_abs*(1 - m_qd_rn_heat[h_rn]),
        e_rn[h_rn] <= qd_max_rn[h_rn] + qd_max_rn_abs*(1 - m_qd_rn_heat[h_rn]),
        e_rn[h_rn] <= qd_max_rn_abs*m_qd_rn_heat[h_rn],
        e_rn[h_rn] >= qd_max_rn[h_rn],
        qd_rn[h_rn] <= e_rn[h_rn]]
    constraints += [
        qd_max_dc[h_dc] <= qd_max_dc_abs*m_qd_dc_heat[h_dc],
        qd_max_dc[h_dc] >= -qd_max_dc_abs*(1 - m_qd_dc_heat[h_dc]),
        e_dc[h_dc] <= qd_max_dc[h_dc] + qd_max_dc_abs*(1 - m_qd_dc_heat[h_dc]),
        e_dc[h_dc] <= qd_max_dc_abs*m_qd_dc_heat[h_dc],
        e_dc[h_dc] >= qd_max_dc[h_dc],
        qd_dc[h_dc] <= e_dc[h_dc]]
    constraints += [qd_rn[h_rn] >= 0]
    constraints += [qd_dc[h_dc] >= 0]

    # < Heat pump constraints (cooling mode) >
    # GSHP power limits
    # constraints += [pc[c_rn] + p_gshp_rn[c_rn] <= s.rn.p_hp_max_cool[c_rn]]
    # GSHP thermal power
    constraints += [q_gshp_rn[c_rn] == cvx.multiply(p_gshp_rn[c_rn], s.rn.cop_cool[c_rn])]
    # Aux heat limits
    constraints += [p_aux_gshp[c_rn] == 0]

    # ASHP power limits
    constraints += [p_ashp[c_dc] <= s.dc.p_hp_max_cool[c_dc]]
    # ASHP thermal power
    constraints += [q_ashp[c_dc] == cvx.multiply(p_ashp[c_dc], s.dc.cop_cool[c_dc])]
    # Aux heat limits
    constraints += [p_aux_ashp[c_dc] == 0]

    # < Heat pump constraints (general) >
    # GSHP thermal capacity limits (bounded by -inf, inf if excluded from config)
    constraints += [qc + q_gshp_rn <= s.rn.q_hp_max_heat, s.rn.q_hp_max_cool <= qc + q_gshp_rn]

    # ASHP thermal capacity limits (bounded by -inf, inf if excluded from config)
    constraints += [q_ashp <= s.dc.q_hp_max_heat, s.dc.q_hp_max_cool <= q_ashp]

    # < Discharging constraints (cooling mode) >
    # Discharging efficiency
    constraints += [eta_d_rn[c_rn]*tk.pd_max_rn == qd_min_rn[c_rn]]
    constraints += [eta_d_dc[c_dc]*tk.pd_max_dc == qd_min_dc[c_dc]]
    # Discharging thermal power, use the big M method to enforce exact bounds
    constraints += [
        qd_min_rn[c_rn] >= -qd_min_rn_abs*(1 - m_qd_rn_cool[c_rn]),
        qd_min_rn[c_rn] <= qd_min_rn_abs*m_qd_rn_cool[c_rn],
        -e_rn[c_rn] >= qd_min_rn[c_rn] - qd_min_rn_abs*m_qd_rn_cool[c_rn],
        -e_rn[c_rn] >= -qd_min_rn_abs*(1 - m_qd_rn_cool[c_rn]),
        -e_rn[c_rn] <= qd_min_rn[c_rn],
        qd_rn[c_rn] >= -e_rn[c_rn]]
    constraints += [
        qd_min_dc[c_dc] >= -qd_min_dc_abs*(1 - m_qd_dc_cool[c_dc]),
        qd_min_dc[c_dc] <= qd_min_dc_abs*m_qd_dc_cool[c_dc],
        -e_dc[c_dc] >= qd_min_dc[c_dc] - qd_min_dc_abs*m_qd_dc_cool[c_dc],
        -e_dc[c_dc] >= -qd_min_dc_abs*(1 - m_qd_dc_cool[c_dc]),
        -e_dc[c_dc] <= qd_min_dc[c_dc],
        qd_dc[c_dc] >= -e_dc[c_dc]]
    constraints += [qd_rn[c_rn] <= 0]
    constraints += [qd_dc[c_dc] <= 0]

    # < Heat pump constraints (zero-load) >
    # Force off to prevent opposite sign heat pump and discharging heat flows
    constraints += [p_gshp_rn[z_rn] == 0]
    constraints += [p_ashp[z_dc] == 0]

    # < Charging constraints >
    constraints += [cvx.multiply(s.rn.cop_cool, pc) <= qc, qc <= cvx.multiply(s.rn.cop_heat, pc)]

    # < Duty cycle constraints >
    # Duty cycle definitions (heating)
    # constraints += [pc[h_rn] == cvx.multiply(dc[h_rn], s.rn.p_hp_max_heat[h_rn])]
    constraints += [p_gshp_rn[h_rn] == cvx.multiply(d_gshp_rn[h_rn], s.rn.p_hp_max_heat[h_rn])]
    constraints += [p_ashp[h_dc] == cvx.multiply(d_ashp[h_dc], s.dc.p_hp_max_heat[h_dc])]

    # Duty cycle definitions (cooling)
    # constraints += [pc[c_rn] == cvx.multiply(dc[c_rn], s.rn.p_hp_max_cool[c_rn])]
    constraints += [p_gshp_rn[c_rn] == cvx.multiply(d_gshp_rn[c_rn], s.rn.p_hp_max_cool[c_rn])]
    constraints += [p_ashp[c_dc] == cvx.multiply(d_ashp[c_dc], s.dc.p_hp_max_cool[c_dc])]

    # Duty cycle definitions (general)
    constraints += [dc >= cvx.maximum(qc/s.rn.p_hp_max_heat/s.rn.cop_heat, qc/s.rn.p_hp_max_cool/s.rn.cop_cool)]
    constraints += [pc >= cvx.maximum(qc/s.rn.cop_heat, qc/s.rn.cop_cool)]

    # Duty cycle definitions (general)
    constraints += [p_aux_gshp == cvx.multiply(d_aux_gshp, s.rn.p_aux_max)]
    constraints += [p_aux_ashp == cvx.multiply(d_aux_ashp, s.dc.p_aux_max)]
    if not (s.rn.p_aux_max != 0).any():
        constraints += [d_aux_gshp == 0]
    if not (s.dc.p_aux_max != 0).any():
        constraints += [d_aux_ashp == 0]
    constraints += [dc <= 1]
    constraints += [d_gshp_rn <= 1]
    constraints += [d_ashp <= 1]
    constraints += [d_aux_gshp <= 1]
    constraints += [d_aux_ashp <= 1]

    # GSHP cannot charge tanks and deliver heating/cooling to ReNEWW simultaneously
    constraints += [dc + d_gshp_rn <= 1]

    # Cannot simultaneously discharge to DC and ReNEWW houses
    # constraints += [
    #     qd_max_rn <= qd_max_rn_abs*x_is_pos,
    #     qd_max_rn >= -qd_max_rn_abs*(1 - x_is_pos),
    #     qd_min_rn >= qd_min_rn_abs*(1 - x_is_pos),
    #     qd_min_rn <= -qd_min_rn_abs*x_is_pos,
    #     e_qd_heat <= qd_max_rn + qd_max_rn_abs*(1 - x_is_pos),
    #     e_qd_heat <= qd_max_rn_abs*x_is_pos,
    #     e_qd_heat >= qd_max_rn,
    #     -e_qd_cool >= qd_min_rn + qd_min_rn_abs*x_is_pos,
    #     -e_qd_cool >= qd_min_rn_abs*(1 - x_is_pos),
    #     -e_qd_cool <= qd_min_rn,
    #     qd_rn + qd_dc <= e_qd_heat,
    #     qd_rn + qd_dc >= -e_qd_cool]
    constraints += [qd_rn[h] + qd_dc[h] <= e_rn[h]]
    constraints += [qd_rn[c] + qd_dc[c] >= -e_rn[c]]

    # < Energy balance constraints >
    if s.comfort_weight == 0:
        # Enforce hard energy balance equality
        constraints += [q_ashp + p_aux_ashp + qd_dc == s.dc.load]
        constraints += [q_gshp_rn + p_aux_gshp + qd_rn == s.rn.load]

    # < Solve problem >
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver=s.solver, QCPDual=0)
    # Return status and whether solution is optimal
    status, success, failure = (prob.status, prob.status == cvx.OPTIMAL, prob.status != cvx.OPTIMAL)

    # < Check status >
    if failure:
        warn(f"The solution to the CVX optimization problem is infeasible, status is "
             f"'{status}'.\nBaseline (no TES) operation has been assumed and state has been propagated "
             f"for this case.")
        # Preserve initial state
        cvx_state = State.initial(s)
        # Set control values using no TES baseline logic
        cvx_control = simulate_no_tes_baseline(s)[1]
        # Propagate initial state through with no control inputs
        for i in range(n):
            cvx_state.update(tk, cvx_control)
    else:
        # Instantiate state with optimal values
        cvx_state = State(x.value[:-1])
        # Instantiate control with optimal values
        cvx_control = Control(pc.value, np.zeros(n), np.zeros(n), p_gshp_rn.value, np.zeros(n), p_ashp.value,
                              p_aux_gshp.value, p_aux_ashp.value, dc.value, np.zeros(n), np.zeros(n), d_gshp_rn.value,
                              np.zeros(n), d_ashp.value, d_aux_gshp.value, d_aux_ashp.value, qc.value, qd_rn.value,
                              qd_dc.value, q_gshp_rn.value, np.zeros(n), q_ashp.value, qd_max_rn.value,
                              qd_min_rn.value, qd_max_dc.value, qd_min_dc.value, eta_d_rn.value, eta_d_dc.value)

    # < Correct qd_max and qd_min >
    calc_disch_limits(cvx_state, cvx_control)

    # < Calculate charging and discharging duty cycle and power >
    calc_duty_cycle_power(cvx_state, cvx_control, tk, s)

    # < Calculate GSHP to DC quantities
    # Any excess of the duty cycle requirement entails GHSP to ReNEWW direct operation
    idx_viol = np.where(np.round(cvx_control.dc + cvx_control.dd_dc + cvx_control.dd_rn, 4) > 1)
    # Violations in heating
    h_rn_viol = np.intersect1d(h_rn, idx_viol)
    # Violations in cooling
    c_rn_viol = np.intersect1d(c_rn, idx_viol)
    # GSHP to DC heat transfer is governed by the lesser of qd_dc and qc
    cvx_control.q_gshp_dc[h_rn_viol] = np.minimum(cvx_control.qd_dc[h_rn_viol], cvx_control.qc[h_rn_viol])
    cvx_control.q_gshp_dc[c_rn_viol] = np.maximum(cvx_control.qd_dc[c_rn_viol], cvx_control.qc[c_rn_viol])
    cvx_control.p_gshp_dc[h_rn_viol] = np.divide(cvx_control.q_gshp_dc[h_rn_viol], s.rn.cop_heat[h_rn_viol])
    cvx_control.p_gshp_dc[c_rn_viol] = np.divide(cvx_control.q_gshp_dc[c_rn_viol], s.rn.cop_cool[c_rn_viol])
    cvx_control.d_gshp_dc[h_rn_viol] = np.divide(cvx_control.p_gshp_dc[h_rn_viol], s.rn.p_hp_max_heat[h_rn_viol])
    cvx_control.d_gshp_dc[c_rn_viol] = np.divide(cvx_control.p_gshp_dc[c_rn_viol], s.rn.p_hp_max_cool[c_rn_viol])

    # Subtract GSHP to DC heat transfer from qd_dc, qc to maintain energy balance, one quantity will go to zero
    cvx_control.qd_dc -= cvx_control.q_gshp_dc
    cvx_control.qc -= cvx_control.q_gshp_dc

    # Recalculate charging behavior
    cvx_control.pc[h_rn_viol] = np.divide(cvx_control.qc[h_rn_viol], s.rn.cop_heat[h_rn_viol])
    cvx_control.pc[c_rn_viol] = np.divide(cvx_control.qc[c_rn_viol], s.rn.cop_cool[c_rn_viol])
    cvx_control.dc[h_rn_viol] = np.divide(cvx_control.pc[h_rn_viol], s.rn.p_hp_max_heat[h_rn_viol])
    cvx_control.dc[c_rn_viol] = np.divide(cvx_control.pc[c_rn_viol], s.rn.p_hp_max_cool[c_rn_viol])
    if (np.round(cvx_control.pc, 4) < 0).any():
        warn(f"The GSHP to ReNEWW simultaneous charging and discharging constraint led to an infeasible charging "
             f"power setpoint at {len(cvx_control.pc[np.round(cvx_control.pc, 4) < 0])} points with a minimum of "
             f"{min(cvx_control.pc)}")

    # Recalculate discharging behavior
    calc_duty_cycle_power(cvx_state, cvx_control, tk)
    if (np.round(cvx_control.pd_dc, 4) < 0).any():
        warn(f"The GSHP to ReNEWW simultaneous charging and discharging constraint led to an infeasible discharging "
             f"power setpoint at {len(cvx_control.pc[np.round(cvx_control.pd_dc, 4) < 0])} points with a minimum of "
             f"{min(cvx_control.pd_dc)}")

    # Mention violations of discharging and duty cycle criteria
    mention_viols(cvx_state, cvx_control, s)

    # Clamp discharging duty cycle and power
    cvx_control.dd_rn = clamp(cvx_control.dd_rn, 1.0)
    cvx_control.dd_dc = clamp(cvx_control.dd_dc, 1.0)

    return cvx_state, cvx_control