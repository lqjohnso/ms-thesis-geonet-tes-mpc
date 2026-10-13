"""
========================================================================================================================
SCRIPT NAME:    dynamics.py
PROJECT:        CHPB-75
AUTHOR(S):      Liam Johnson
AFFILIATION:    Purdue University, CHPB Herrick Labs
DATE CREATED:   8-16-2026
LAST MODIFIED:  10-06-2026
VERSION:        1.4
PYTHON VERSION: 3.14.6
========================================================================================================================
DESCRIPTION: This module defines the State and Control classes which are used to update the state of the tank throughout
simulations and record results.

DEPENDENCIES:
    - coolprop
    - numpy
    - pandas
    - python-openei
    - requests
    - scipy
    - toml
========================================================================================================================
"""

####################
# Required Imports #
####################

from build_models import Models
from copy import deepcopy
from models import Tank
import numpy as np
import pandas as pd
from scenario import Scenario
from utils import calc_duty_cycle
from utils import clamp
from utils import count_init_args
from utils import reindex_cls
from utils import safe_div

###########
# Classes #
###########


class State:
    """A class representing the TES tank state.

    Attributes:
        x (np.ndarray): 1-D array of stored tank thermal energy in kWh
    """
    def __init__(self, x: np.ndarray):
        """Initialize attributes of the State class.

        Parameters:
            x: A 1-D array of the stored tank thermal energy in kWh
        """

        # < Assign class attributes >
        self.x = x
        # Private counter attribute indicating index of last calculated state
        self.__count = 0

    @classmethod
    def initial(cls, scenario: Scenario):
        """Factory method for construction by filling arrays with NaN except for index 0 which is the initial state
            provided in a scenario instance.

        Parameters:
            scenario: An instance of the Scenario class from which the initial state is derived.
        """

        # < Assign __init__ parameters >
        if isinstance(scenario.t, pd.DatetimeIndex):
            x = np.full(len(scenario.t), np.nan)
        else:
            x = np.full(1, np.nan)
        if x[0] is not None:
            x[0] = scenario.x_0

        return cls(x)

    @classmethod
    def blank(cls, scenario: Scenario | None = None, n: int | None = None, fill_val: float | np.nan | None = None):
        """Factory method for construction by filling arrays with NaN or another fill value.

        Parameters:
            scenario: An instance of the Scenario class from which the length of the simulation (in timesteps) is derived
            n: The length of the simulation (in timesteps)
            fill_val: Value to fill array with
        """

        # < Validate params >
        assert (scenario is not None) or (n is not None), f"Either scenario or n must be provided"

        # < Set default params >
        if fill_val is None:
            fill_val = np.nan
        elif isinstance(fill_val, int):
            fill_val = float(fill_val)

        # < Condition params >
        if scenario is not None:
            n = len(scenario.t)

        # < Assign __init__ parameters >
        fill_arrs = [np.full(n, fill_val) for _ in range(count_init_args(cls))]
        return cls(*fill_arrs)

    def update(self, tank: Tank, control: Control):
        """Updates the tank state given the control input, filling in the next element in the x array."""
        k = self.__count
        x_k = self.x[k]
        x_end = self.x[-1]
        qc = control.qc[k]
        qd_rn = control.qd_rn[k]
        qd_dc = control.qd_dc[k]

        # Do not calculate the state that occurs one timestep after the simulation
        if (np.isnan(x_end) and (not np.isnan(x_k))) or ((x_end is None) and (x_k is not None)):
            # Tank dynamics
            x_k1 = tank.a*x_k + tank.b*(qc - (qd_rn + qd_dc))
            # Clamp new state within tank limits and assign
            self.x[k + 1] = clamp(x_k1, tank.x_max, tank.x_min)
        self.__count += 1

    def fill(self, x):
        """Fills an already instantiated class with an x array."""

        if len(x) != len(self.x):
            self.x = self.x[:-2]
        else:
            self.x = x

    @property
    def now(self) -> float:
        """Returns the current state of the tank."""

        if isinstance(self.x, np.ndarray):
            return self.x[self.__count]
        else:
            return self.x


class Control:
    """A class representing the control inputs which influence the tank state.

    Attributes:
        pc (np.ndarray): 1-D array of GSHP electrical power used for TES charging in kW
        pd_rn (np.ndarray): 1-D array of electrical pumping power used to discharge thermal storage to the ReNEWW House
            in kW
        pd_dc (np.ndarray): 1-D array of electrical pumping power used to discharge thermal storage to the DC House in
            kW
        p_gshp_rn (np.ndarray): 1-D array of GSHP electrical power used to meet the thermal load in the ReNEWW House in
            kW
        p_gshp_dc (np.ndarray): 1-D array of GSHP electrical power used to meet the thermal load in the DC House in kW
        p_ashp (np.ndarray): 1-D array of ASHP electrical power used to meet the thermal load in the DC House in kW
        p_aux_gshp (np.ndarray): 1-D array of GSHP auxiliary power used to meet the thermal load in the ReNEWW House in kW
        p_aux_ashp (np.ndarray): 1-D array of GSHP auxiliary power used to meet the thermal load in the DC House in kW
        dc (np.ndarray): 1-D array of GSHP duty cycle (0-1) used for TES charging
        dd_rn (np.ndarray): 1-D array of pumping duty cycle (0-1) used to discharge thermal storage to the ReNEWW House
        dd_dc (np.ndarray): 1-D array of pumping duty cycle (0-1) used to discharge thermal storage to the DC House in
            kW
        d_gshp_rn (np.ndarray): 1-D array of GSHP duty cycle (0-1) used to meet the thermal load in the ReNEWW House in
            kW
        d_gshp_dc (np.ndarray): 1-D array of GSHP duty cycle (0-1) used to meet the thermal load in the DC House in kW
        d_ashp (np.ndarray): 1-D array of ASHP duty cycle (0-1) used to meet the thermal load in the DC House
        d_aux_gshp (np.ndarray): 1-D array of GSHP auxiliary duty cycle (0-1) used to meet the thermal load in the
            ReNEWW House
        d_aux_ashp (np.ndarray): 1-D array of GSHP auxiliary duty cycle (0-1) used to meet the thermal load in the DC
            House
        qc (np.ndarray): 1-D array of GSHP thermal power used for TES charging in kW
        qd_rn (np.ndarray): 1-D array of TES discharging thermal power to the ReNEWW House in kW
        qd_dc (np.ndarray): 1-D array of TES discharging thermal power to the DC House in kW
        q_gshp_rn (np.ndarray): 1-D array of GSHP thermal power used to meet the thermal load in the ReNEWW House in kW
        q_gshp_dc (np.ndarray): 1-D array of GSHP thermal power used to meet the thermal load in the DC House in kW
        q_ashp (np.ndarray): 1-D array of ASHP thermal power used to meet the thermal load in the DC House in kW
        q_tot_del_rn (np.ndarray): 1-D array of total power delivered to the ReNEWW House in kW
        q_tot_del_dc (np.ndarray): 1-D array of total power delivered to the DC House in kW
        qd_max_rn (np.ndarray): 1-D array of the maximum discharging thermal power possible to the ReNEWW House in kW
            (stored in Control because it is a function of stored tank energy, not deterministic)
        qd_min_rn (np.ndarray): 1-D array of the minimum discharging thermal power possible to the ReNEWW House in kW
            (stored in Control because it is a function of stored tank energy, not deterministic)
        qd_max_dc (np.ndarray): 1-D array of the maximum discharging thermal power possible to the DC House in kW
            (stored in Control because it is a function of stored tank energy, not deterministic)
        qd_min_dc (np.ndarray): 1-D array of the minimum discharging thermal power possible to the DC House in kW
            (stored in Control because it is a function of stored tank energy, not deterministic)
        eta_d_rn (np.ndarray): 1-D array of the efficiency of TES discharging to the ReNEWW House
        eta_d_dc (np.ndarray): 1-D array of the efficiency of TES discharging to the DC House
    """

    def __init__(self, pc: np.ndarray, pd_rn: np.ndarray, pd_dc: np.ndarray, p_gshp_rn: np.ndarray,
                 p_gshp_dc: np.ndarray, p_ashp: np.ndarray, p_aux_gshp: np.ndarray, p_aux_ashp: np.ndarray,
                 dc: np.ndarray, dd_rn: np.ndarray, dd_dc: np.ndarray, d_gshp_rn: np.ndarray,
                 d_gshp_dc: np.ndarray, d_ashp: np.ndarray, d_aux_gshp: np.ndarray, d_aux_ashp: np.ndarray,
                 qc: np.ndarray, qd_rn: np.ndarray, qd_dc: np.ndarray, q_gshp_rn: np.ndarray, q_gshp_dc: np.ndarray,
                 q_ashp: np.ndarray, qd_max_rn: np.ndarray, qd_min_rn: np.ndarray, qd_max_dc: np.ndarray,
                 qd_min_dc: np.ndarray, eta_d_rn: np.ndarray | None = None, eta_d_dc: np.ndarray | None = None):
        """Initialize attributes of the Control class.

        Parameters:
            pc: 1-D array of GSHP electrical power used for TES charging in kW
            pd_rn: 1-D array of electrical pumping power used to discharge thermal storage to the ReNEWW House
                in kW
            pd_dc: 1-D array of electrical pumping power used to discharge thermal storage to the DC House in
                kW
            p_gshp_rn: 1-D array of GSHP electrical power used to meet the thermal load in the ReNEWW House in
                kW
            p_gshp_dc: 1-D array of GSHP electrical power used to meet the thermal load in the DC House in kW
            p_ashp: 1-D array of ASHP electrical power used to meet the thermal load in the DC House in kW
            p_aux_gshp: 1-D array of GSHP auxiliary power used to meet the thermal load in the ReNEWW House in kW
            p_aux_ashp: 1-D array of GSHP auxiliary power used to meet the thermal load in the DC House in kW
            dc: 1-D array of GSHP duty cycle (0-1) used for TES charging
            dd_rn: 1-D array of pumping duty cycle (0-1) used to discharge thermal storage to the ReNEWW House
            dd_dc: 1-D array of pumping duty cycle (0-1) used to discharge thermal storage to the DC House in kW
            d_gshp_rn: 1-D array of GSHP duty cycle (0-1) used to meet the thermal load in the ReNEWW House in kW
            d_ashp: 1-D array of ASHP duty cycle (0-1) used to meet the thermal load in the DC House
            d_aux_gshp: 1-D array of GSHP auxiliary duty cycle (0-1) used to meet the thermal load in the ReNEWW House
            d_aux_ashp: 1-D array of GSHP auxiliary duty cycle (0-1) used to meet the thermal load in the DC House
            qc: 1-D array of GSHP thermal power used for TES charging in kW
            qd_rn: 1-D array of TES discharging thermal power to the ReNEWW House in kW
            qd_dc: 1-D array of TES discharging thermal power to the DC House in kW
            q_gshp_rn: 1-D array of GSHP thermal power used to meet the thermal load in the ReNEWW House in kW
            d_gshp_dc: 1-D array of GSHP duty cycle (0-1) used to meet the thermal load in the DC House in kW
            q_gshp_dc: 1-D array of GSHP thermal power used to meet the thermal load in the DC House in kW
            q_ashp: 1-D array of ASHP thermal power used to meet the thermal load in the DC House in kW
            q_tot_del_rn: 1-D array of total power delivered to the ReNEWW House in kW
            q_tot_del_dc: 1-D array of total power delivered to the DC House in kW
            qd_max_rn: 1-D array of the maximum discharging thermal power possible to the ReNEWW House in kW
                (stored in Control because it is a function of stored tank energy, not deterministic)
            qd_min_rn: 1-D array of the minimum discharging thermal power possible to the ReNEWW House in kW
                (stored in Control because it is a function of stored tank energy, not deterministic)
            qd_max_dc: 1-D array of the maximum discharging thermal power possible to the DC House in kW
                (stored in Control because it is a function of stored tank energy, not deterministic)
            qd_min_dc: 1-D array of the minimum discharging thermal power possible to the DC House in kW
                (stored in Control because it is a function of stored tank energy, not deterministic)
            eta_d_rn: 1-D array of the efficiency of TES discharging to the ReNEWW House
            eta_d_dc: 1-D array of the efficiency of TES discharging to the DC House
    """

        # < Calculate class attributes >
        q_tot_del_rn = qd_rn + q_gshp_rn + p_aux_gshp
        q_tot_del_dc = qd_dc + q_ashp + p_aux_ashp + q_gshp_dc

        # < Assign class attributes >
        self.pc = pc
        self.pd_rn = pd_rn
        self.pd_dc = pd_dc
        self.p_gshp_rn = p_gshp_rn
        self.p_gshp_dc = p_gshp_dc
        self.p_ashp = p_ashp
        self.p_aux_gshp = p_aux_gshp
        self.p_aux_ashp = p_aux_ashp
        self.dc = dc
        self.dd_rn = dd_rn
        self.dd_dc = dd_dc
        self.d_gshp_rn = d_gshp_rn
        self.d_gshp_dc = d_gshp_dc
        self.d_ashp = d_ashp
        self.d_aux_gshp = d_aux_gshp
        self.d_aux_ashp = d_aux_ashp
        self.qc = qc
        self.qd_rn = qd_rn
        self.qd_dc = qd_dc
        self.q_gshp_rn = q_gshp_rn
        self.q_gshp_dc = q_gshp_dc
        self.q_ashp = q_ashp
        self.q_tot_del_rn = q_tot_del_rn
        self.q_tot_del_dc = q_tot_del_dc
        self.qd_max_rn = qd_max_rn
        self.qd_min_rn = qd_min_rn
        self.qd_max_dc = qd_max_dc
        self.qd_min_dc = qd_min_dc
        self.eta_d_rn = eta_d_rn
        self.eta_d_dc = eta_d_dc
        # Private counter attribute indicating index of last clipped control input
        self.__count = 0

    @classmethod
    def blank(cls, scenario: Scenario | None = None, n: int | None = None, fill_val: float | np.nan | None = None):
        """Factory method for construction by filling arrays with NaN or another fill value.

        Parameters:
            scenario: An instance of the Scenario class from which the length of the simulation (in timesteps) is derived
            n: The length of the simulation (in timesteps)
            fill_val: Value to fill array with
        """
        # < Validate params >
        assert (scenario is not None) or (n is not None), f"Either scenario or n must be provided"

        # < Set default params >
        if fill_val is None:
            fill_val = np.nan
        elif isinstance(fill_val, int):
            fill_val = float(fill_val)

        # < Condition params >
        if scenario is not None:
            n = len(scenario.t)

        # < Assign __init__ parameters >
        fill_arrs = [np.full(n, fill_val) for _ in range(count_init_args(cls))]
        return cls(*fill_arrs)

    def insert(self, u: Control):
        """Insert an instance u with 0-D attributes into the next available array index."""

        k = self.__count

        self.pc[k] = u.pc
        self.pd_rn[k] = u.pd_rn
        self.pd_dc[k] = u.pd_dc
        self.p_gshp_rn[k] = u.p_gshp_rn
        self.p_gshp_dc[k] = u.p_gshp_dc
        self.p_ashp[k] = u.p_ashp
        self.p_aux_gshp[k] = u.p_aux_gshp
        self.p_aux_ashp[k] = u.p_aux_ashp
        self.dc[k] = u.dc
        self.dd_rn[k] = u.dd_rn
        self.dd_dc[k] = u.dd_dc
        self.d_gshp_rn[k] = u.d_gshp_rn
        self.d_gshp_dc[k] = u.d_gshp_dc
        self.d_ashp[k] = u.d_ashp
        self.d_aux_gshp[k] = u.d_aux_gshp
        self.d_aux_ashp[k] = u.d_aux_ashp
        self.qc[k] = u.qc
        self.qd_rn[k] = u.qd_rn
        self.qd_dc[k] = u.qd_dc
        self.q_gshp_rn[k] = u.q_gshp_rn
        self.q_gshp_dc[k] = u.q_gshp_dc
        self.q_ashp[k] = u.q_ashp
        self.q_tot_del_rn[k] = u.q_tot_del_rn
        self.q_tot_del_dc[k] = u.q_tot_del_dc
        self.qd_max_rn[k] = u.qd_max_rn
        self.qd_min_rn[k] = u.qd_min_rn
        self.qd_max_dc[k] = u.qd_max_dc
        self.qd_min_dc[k] = u.qd_min_dc
        self.eta_d_rn[k] = u.eta_d_rn
        self.eta_d_dc[k] = u.eta_d_dc

        self.__count += 1

    def clip(self, control_horiz: Control, scenario: Scenario, m: Models, state: State):
        """Clips first control inputs control_horiz so they are bounded by real conditions and system behavior, then
        adds them to arrays in self. Follows the order of operations below:

        discharge -> GSHP to ReNEWW and ASHP to DC -> GSHP to DC -> charge

        Parameters:
            control_horiz: An instance of the Control class for the MPC horizon
            scenario: An instance of the Scenario class
            m: An instance of the Models class
            state: An instance of the State class used in the MPC simulation
        """

        k = self.__count
        # Take only the first control input
        u = deepcopy(control_horiz)
        reindex_cls(u, 0, as_scalar=True)
        # Use scenario parameters at correct timestep
        s = deepcopy(scenario)
        reindex_cls(s, k, as_scalar=True, exclude=['controllers'])
        x = state.now

        # < Clipping >
        # Discharging efficiency
        pd_max_rn = m.tank_real.pd_max_rn
        pd_max_dc = m.tank_real.pd_max_dc
        # Enforce cycling constraint, cannot simultaneously discharge to ReNEWW and DC Houses (DC gets priority)
        u.dd_rn = clamp(u.dd_rn, 1 - u.dd_dc, 0)
        # Use duty cycle and real limits to get realistic power setpoints
        u.pd_rn = u.dd_rn*pd_max_rn
        u.pd_dc = u.dd_dc*pd_max_dc

        if x > 0:
            # < Tank heating potential >
            # Use qd_max_rn, pd_max_rn and clamp so result is non-negative
            u.qd_max_rn = np.polyval(m.tank_real.qd_max_rn, x)
            u.qd_min_rn = 0
            u.qd_max_dc = np.polyval(m.tank_real.qd_max_dc, x)
            u.qd_min_dc = 0
            if u.qd_max_rn < 0:
                # Ensure limits are correct when there is no heating or cooling potential
                u.qd_max_rn = 0
            if u.qd_max_dc < 0:
                u.qd_max_dc = 0
            u.eta_d_rn = clamp(safe_div(u.qd_max_rn, pd_max_rn, 0), min_value=0)
            u.eta_d_dc = clamp(safe_div(u.qd_max_dc, pd_max_dc, 0), min_value=0)
        else:
            # < Tank cooling potential >
            # Use qd_min_rn, pd_min_rn and clamp so result is non-positive
            u.qd_max_rn = 0
            u.qd_min_rn = np.polyval(m.tank_real.qd_min_rn, x)
            u.qd_max_dc = 0
            u.qd_min_dc = np.polyval(m.tank_real.qd_min_dc, x)
            if u.qd_min_rn > 0:
                # Ensure limits are correct when there is no heating or cooling potential
                u.qd_min_rn = 0
            if u.qd_min_dc > 0:
                u.qd_min_dc = 0
            u.eta_d_rn = clamp(safe_div(u.qd_min_rn, pd_max_rn, 0), max_value=0)
            u.eta_d_dc = clamp(safe_div(u.qd_min_dc, pd_max_dc, 0), max_value=0)

        # Discharging electrical power
        # Ensure discharging power is within limits and will match house thermal load instead of exceeding
        # Use delta to improve numerical stability
        delta = .00001
        u.pd_rn = min(clamp(u.pd_rn, pd_max_rn, 0),
                      clamp(safe_div(s.rn.load_real, u.eta_d_rn + delta*np.sign(u.eta_d_rn), 0), min_value=0))
        u.pd_dc = min(clamp(u.pd_dc, pd_max_dc, 0),
                      clamp(safe_div(s.dc.load_real, u.eta_d_dc + delta*np.sign(u.eta_d_dc), 0), min_value=0))
        # Update duty cycle
        u.dd_rn = calc_duty_cycle(u.pd_rn, pd_max_rn)
        u.dd_dc = calc_duty_cycle(u.pd_dc, pd_max_dc)
        # Discharging thermal power
        u.qd_rn = u.eta_d_rn*u.pd_rn
        u.qd_dc = u.eta_d_dc*u.pd_dc

        if s.rn.load_real > 0:
            # < ReNEWW House heating mode >
            # Use duty cycle to get updated GSHP to ReNEWW setpoint
            u.p_gshp_rn = u.d_gshp_rn*s.rn.p_hp_max_heat_real
            # GSHP will be used to meet thermal load not met by discharging
            u.p_gshp_rn = (s.rn.load_real - u.qd_rn)/s.rn.cop_heat_real
            # Keep within power limits
            u.p_gshp_rn = clamp(u.p_gshp_rn, s.rn.p_hp_max_heat_real, 0)
            # Keep within capacity limits
            u.p_gshp_rn = clamp(u.p_gshp_rn, s.rn.q_hp_max_heat_real/s.rn.cop_heat_real, 0)
            # Update duty cycle
            u.d_gshp_rn = calc_duty_cycle(u.p_gshp_rn, s.rn.p_hp_max_heat_real)
            # Calculate thermal power under real O.C.'s
            u.q_gshp_rn = u.p_gshp_rn*s.rn.cop_heat_real
            # Use aux heat for remainder of thermal load
            u.p_aux_gshp = s.rn.load_real - (u.q_gshp_rn + u.qd_rn)
            u.p_aux_gshp = clamp(u.p_aux_gshp, m.gshp.p_aux_max, 0)
            # Update duty cycle
            u.d_aux_gshp = calc_duty_cycle(u.p_aux_gshp, m.gshp.p_aux_max)

            # Use duty cycle to get updated GSHP to DC setpoint
            u.p_gshp_dc = u.d_gshp_dc*s.rn.p_hp_max_heat_real
            # GSHP to DC electrical power
            # Ensure power is within limits and will match house thermal load instead of exceeding
            u.p_gshp_dc = min(clamp(u.p_gshp_dc, s.rn.p_hp_max_heat_real - u.p_gshp_rn, 0),
                              clamp(safe_div(s.dc.load_real - u.qd_dc, s.rn.cop_heat_real + delta, 0), min_value=0))
            # Update duty cycle
            u.d_gshp_dc = calc_duty_cycle(u.p_gshp_dc, s.rn.p_hp_max_heat_real)
            # Calculate thermal power under real O.C.'s
            u.q_gshp_dc = u.p_gshp_dc*s.rn.cop_heat_real
        else:
            # < ReNEWW House cooling mode >
            # Use duty cycle to get updated setpoint
            u.p_gshp_rn = u.d_gshp_rn*s.rn.p_hp_max_cool_real
            # GSHP will be used to meet thermal load not met by discharging
            u.p_gshp_rn = (s.rn.load_real - u.qd_rn)/s.rn.cop_cool_real
            # Keep within power limits
            u.p_gshp_rn = clamp(u.p_gshp_rn, s.rn.p_hp_max_cool_real, 0)
            # Keep within capacity limits
            u.p_gshp_rn = clamp(u.p_gshp_rn, s.rn.q_hp_max_cool_real/s.rn.cop_cool_real, 0)
            # Update duty cycle
            u.d_gshp_rn = calc_duty_cycle(u.p_gshp_rn, s.rn.p_hp_max_cool_real)
            # Calculate thermal power under real O.C.'s
            u.q_gshp_rn = u.p_gshp_rn*s.rn.cop_cool_real
            # No aux heat can be used
            u.p_aux_gshp = 0
            u.d_aux_gshp = 0

            # Use duty cycle to get updated GSHP to DC setpoint
            u.p_gshp_dc = u.d_gshp_dc*s.rn.p_hp_max_cool_real
            # GSHP to DC electrical power
            # Ensure power is within limits and will match house thermal load instead of exceeding
            u.p_gshp_dc = min(clamp(u.p_gshp_dc, s.rn.p_hp_max_cool_real - u.p_gshp_dc, 0),
                              clamp(safe_div(s.dc.load_real - u.qd_dc, s.rn.cop_cool_real - delta, 0), min_value=0))
            # Update duty cycle
            u.d_gshp_dc = calc_duty_cycle(u.p_gshp_dc, s.rn.p_hp_max_cool_real)
            # Calculate thermal power under real O.C.'s
            u.q_gshp_dc = u.p_gshp_dc*s.rn.cop_cool_real

        if s.dc.load_real > 0:
            # < DC House heating mode >
            # Use duty cycle to get updated setpoint
            u.p_ashp = u.d_ashp*s.dc.p_hp_max_heat_real
            # ASHP will be used to meet thermal load not met by discharging or GSHP
            u.p_ashp = (s.dc.load_real - u.qd_dc - u.q_gshp_dc)/s.dc.cop_heat_real
            # Keep within power limits
            u.p_ashp = clamp(u.p_ashp, s.dc.p_hp_max_heat_real, 0)
            # Keep within capacity limits
            u.p_ashp = clamp(u.p_ashp, s.dc.q_hp_max_heat_real/s.dc.cop_heat_real, 0)
            # Update duty cycle
            u.d_ashp = calc_duty_cycle(u.p_ashp, s.dc.p_hp_max_heat_real)
            # Calculate thermal power under real O.C.'s
            u.q_ashp = u.p_ashp*s.dc.cop_heat_real
            # Use aux heat for remainder of thermal load
            u.p_aux_ashp = s.dc.load_real - (u.q_ashp + u.qd_dc + u.q_gshp_dc)
            u.p_aux_ashp = clamp(u.p_aux_ashp, m.ashp.p_aux_max, 0)
            # Update duty cycle
            u.d_aux_ashp = calc_duty_cycle(u.p_aux_ashp, m.ashp.p_aux_max)
        else:
            # < DC House cooling mode >
            # Use duty cycle to get updated setpoint
            u.p_ashp = u.d_ashp*s.dc.p_hp_max_cool_real
            # ashp will be used to meet thermal load not met by discharging
            u.p_ashp = (s.dc.load_real - u.qd_dc - u.q_gshp_dc)/s.dc.cop_cool_real
            # Keep within power limits
            u.p_ashp = clamp(u.p_ashp, s.dc.p_hp_max_cool_real, 0)
            # Keep within capacity limits
            u.p_ashp = clamp(u.p_ashp, s.dc.q_hp_max_cool_real/s.dc.cop_cool_real, 0)
            # Update duty cycle
            u.d_ashp = calc_duty_cycle(u.p_ashp, s.dc.p_hp_max_cool_real)
            # Calculate thermal power under real O.C.'s
            u.q_ashp = u.p_ashp*s.dc.cop_cool_real
            # No aux heat can be used
            u.p_aux_ashp = 0
            u.d_aux_ashp = 0

        # Charging electrical power
        # Available duty cycle for charging
        dc_avail = min(1 - (u.d_gshp_rn + u.d_gshp_dc), u.dc, 1 - (u.dd_rn + u.dd_dc))
        # Enforce cycling constraints, cannot simultaneously use GSHP to meet ReNEWW/DC thermal load and charge tanks
        # or to charge and discharge
        u.dc = clamp(u.dc, dc_avail, 0)
        if u.qc > 0:
            # < If charging tank for heating >
            u.pc = u.dc*s.rn.p_hp_max_heat_real
            # Keep within charging limits
            u.pc = clamp(u.pc, s.rn.p_hp_max_heat_real - u.p_gshp_rn - u.p_gshp_dc, 0)
            # Keep within capacity limits
            if s.rn.q_hp_max_heat_real != np.inf:
                u.pc = clamp(u.pc, s.rn.q_hp_max_heat_real*dc_avail/s.rn.cop_heat_real, 0)
            # Update duty cycle
            u.dc = calc_duty_cycle(u.pc, s.rn.p_hp_max_heat_real)
            u.qc = u.pc*s.rn.cop_heat_real
        else:
            # < If charging tank for cooling >
            u.pc = u.dc*s.rn.p_hp_max_cool_real
            # Keep within charging limits
            u.pc = clamp(u.pc, s.rn.p_hp_max_cool_real - u.p_gshp_rn - u.p_gshp_dc, 0)
            # Keep within capacity limits
            if s.rn.q_hp_max_cool_real != -np.inf:
                u.pc = clamp(u.pc, s.rn.q_hp_max_cool_real*dc_avail/s.rn.cop_cool_real, 0)
            # Update duty cycle
            u.dc = calc_duty_cycle(u.pc, s.rn.p_hp_max_cool_real)
            u.qc = u.pc*s.rn.cop_cool_real

        # Total power
        u.q_tot_del_rn = u.qd_rn + u.q_gshp_rn + u.p_aux_gshp
        u.q_tot_del_dc = u.qd_dc + u.q_ashp + u.p_aux_ashp + u.q_gshp_dc

        # Insert clipped values from first horizon timestep
        self.insert(u)
