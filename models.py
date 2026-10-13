"""
========================================================================================================================
SCRIPT NAME:    models.py
PROJECT:        CHPB-75
AUTHOR(S):      Liam Johnson
AFFILIATION:    Purdue University, CHPB Herrick Labs
DATE CREATED:   7-20-2026
LAST MODIFIED:  10-06-2026
VERSION:        1.3
PYTHON VERSION: 3.14.6
========================================================================================================================
DESCRIPTION: This module defines the classes representing the tank and heat pump models used in the simulation.

DEPENDENCIES:
    - coolprop
    - numpy
    - pandas
    - scipy
========================================================================================================================
"""

####################
# Required Imports #
####################

from CoolProp.CoolProp import PropsSI
import numpy as np
import pandas as pd
from typing import Literal
from utils import make_1d
from utils import make_array
from utils import mov_mean
from utils import set_timedelta
from utils import smoothdata

##################
# Model Classes #
##################


class Tank:
    """A class representing a lumped capacitance RC TES tank model.

    Attributes:
        a (float): Discrete-time dynamics coefficient
        b (float): Discrete-time dynamics coefficient
        x_max (float): Maximum stored tank thermal energy in kWh
        x_min (float): Minimum stored tank thermal energy in kWh
        pd_max_rn (float): Maximum electrical pumping power used to discharge thermal storage to the ReNEWW House in kW
        pd_max_dc (float): Maximum electrical pumping power used to discharge thermal storage to the DC House in kW
        qd_max_rn (np.ndarray): 1-D array containing the coefficients of a linear polynomial model for maximum discharging
            thermal power to the ReNEWW House in kW from tank stored thermal energy in kWh (used when tank has heating
            potential)
        qd_min_rn (np.ndarray): 1-D array containing the coefficients of a linear polynomial model for minimum discharging
            thermal power to the ReNEWW House in kW from tank stored thermal energy in kWh (used when tank has cooling
            potential)
        qd_max_dc (np.ndarray): 1-D array containing the coefficients of a linear polynomial model for maximum discharging
            thermal power to the DC House in kW from tank stored thermal energy in kWh (used when tank has heating
            potential)
        qd_min_dc (np.ndarray): 1-D array containing the coefficients of a linear polynomial model for minimum discharging
            thermal power to the DC House in kW from tank stored thermal energy in kWh (used when tank has cooling
            potential)
    """

    @staticmethod
    def __qd_linfit(qd: np.ndarray, x_qd: float, x_int: float) -> np.ndarray:
        """Fits a linear polynomial model for the TES discharging power in kW with respect to the stored tank energy in
        kWh if qd does not already contain a coefficient array.

        Parameters:
            qd: A qd_max or qd_min value in kW initially inputted to __init__ in the Tank class
            x_qd: Location of qd on the x-axis in kWh
            x_int: Location of the x-intercept in kWh

        Output:
            qd: A qd_max or qd_min coefficient array
        """
        assert isinstance(qd, np.ndarray), f"qd must be np.ndarray."
        if len(qd) == 1:
            qd = np.polyfit([x_int, x_qd], [0, qd.item()], 1)
        return qd

    def __init__(
            self,
            dt: float,
            temp_max_heat: float,
            temp_min_cool: float,
            pd_max_rn: float,
            pd_max_dc: float,
            qd_max_rn: float | list | np.ndarray | None = None,
            qd_min_rn: float | list | np.ndarray | None = None,
            qd_max_dc: float | list | np.ndarray | None = None,
            qd_min_dc: float | list | np.ndarray | None = None,
            temp_amb: float | None = None,
            temp_min_heat: float | None = None,
            temp_max_cool: float | None = None,
            q_loss: float | None = None,
            temp_ref_q_loss: float | None = None,
            n_tanks: int | None = None,
            h_tank: float | None = None,
            d_tank: float | None = None,
            v_tank: float | None = None,
            p_tank: float | None = None,
            k_ins: float | None = None,
            l_ins: float | None = None,
            fluid: str | None = None,
            r_tank: float | None = None,
            c_tank: float | None = None):
        """Initialize attributes of the Tank class.

        Parameters:
            dt: The simulation timestep in h
            temp_max_heat: Maximum achievable tank temperature for heating mode in degC
            temp_min_cool: Minimum tank temperature for cooling mode in degC
            pd_max_rn: Maximum electrical pumping power used to discharge thermal storage to the ReNEWW House in kW
            pd_max_dc: Maximum electrical pumping power used to discharge thermal storage to the DC House in kW
                qd_max_rn: 1-D array containing the coefficients of a linear polynomial model for maximum discharging
                thermal power to the ReNEWW House in kW from tank stored thermal energy in kWh (used when tank has
                heating potential)
            qd_min_rn: 1-D array containing the coefficients of a linear polynomial model for minimum discharging
                thermal power to the ReNEWW House in kW from tank stored thermal energy in kWh (used when tank has
                cooling potential)
            qd_max_dc: 1-D array containing the coefficients of a linear polynomial model for maximum discharging
                thermal power to the DC House in kW from tank stored thermal energy in kWh (used when tank has heating
                potential) or float representing the maximum discharging thermal power at temp_max_heat
            qd_min_dc: 1-D array containing the coefficients of a linear polynomial model for minimum discharging
                thermal power to the DC House in kW from tank stored thermal energy in kWh (used when tank has cooling
                potential)
            temp_amb: Ambient temperature with respect to the tank in degC
            temp_min_heat: Minimum tank temperature for heating mode in degC
            temp_max_cool: Maximum achievable tank temperature for cooling mode in degC
            q_loss: Rate of passive heat loss (positive) or heat gain (negative) for each tank at temp_ref_q_loss in kW
            temp_ref_q_loss: Reference tank temperature at which q_loss occurs in degC
            n_tanks : Number of TES tanks available
            h_tank : Height of each tank in m
            d_tank : Diameter of each tank in m
            v_tank: Fluid volumetric capacity of each tank in m^3
            p_tank: Fluid pressure in each tank in kPa
            k_ins: Thermal conductivity of insulation in W/m-k
            l_ins = Thickness of thermal insulation in m
            fluid: coolprop fluid or mixture used in the system
            r_tank: Thermal resistance from tank internal temperature to the ambient in degC/kW
            c_tank: Thermal capacitance of the tank in kWh/degC
        """

        # < Set default params >
        if temp_amb is None:
            temp_amb = 20.0
        if temp_min_heat is None:
            temp_min_heat = temp_amb
        if temp_max_cool is None:
            temp_max_cool = temp_amb
        if qd_max_rn is None:
            qd_max_rn = 0.0
        if qd_min_rn is None:
            qd_min_rn = 0.0
        if qd_max_dc is None:
            qd_max_dc = 0.0
        if qd_min_dc is None:
            qd_min_dc = 0.0
        if temp_ref_q_loss is None:
            if (q_loss is not None) and (q_loss > 0):
                temp_ref_q_loss = (temp_max_heat + temp_min_heat)/2
            else:
                temp_ref_q_loss = (temp_max_cool + temp_min_cool)/2
        if p_tank is None:
            p_tank = 1013.25
        if fluid is None:
            fluid = "Water"

        # < Condition params >
        dt = round(dt*60)/60
        qd_max_rn = make_1d(qd_max_rn)
        qd_min_rn = make_1d(qd_min_rn)
        qd_max_dc = make_1d(qd_max_dc)
        qd_min_dc = make_1d(qd_min_dc)

        # < Validate params >
        assert dt > 0, f"dt must be greater than 0, got {dt}."
        assert pd_max_dc >= 0, f"pd_max_dc must be greater than or equal to 0, got {pd_max_dc}."
        assert pd_max_rn >= 0, f"pd_max_dc must be greater than or equal to 0, got {pd_max_rn}."
        assert len(qd_max_rn) <= 2, (f"qd_max_rn must be linear or scalar, higher order models break the convexity of "
                                     f"the optimization.")
        assert len(qd_min_rn) <= 2, (f"qd_min_rn must be linear or scalar, higher order models break the convexity of "
                                     f"the optimization.")
        assert len(qd_max_dc) <= 2, (f"qd_max_dc must be linear or scalar, higher order models break the convexity of "
                                     f"the optimization.")
        assert len(qd_min_dc) <= 2, (f"qd_min_dc must be linear or scalar, higher order models break the convexity of "
                                     f"the optimization.")
        assert (pd_max_rn != 0) or (pd_max_dc != 0), (f"pd_max_rn or pd_max_dc must be nonzero, or else no TES "
                                                      f"discharging is possible.")
        assert (qd_max_rn != 0) or (qd_min_rn != 0) or (qd_max_dc != 0) or (qd_min_dc != 0), \
            f"qd_max_rn, qd_min_rn, qd_max_dc, or qd_min_dc must be nonzero, or else no TES discharging is possible."
        assert temp_max_heat > temp_min_heat, (f"temp_max_heat must be greater than temp_min_heat, got {temp_max_heat} "
                                               f"and {temp_min_heat} respectively.")
        assert temp_min_heat >= temp_amb, (f"temp_min_heat must be greater than or equal to temp_amb, got "
                                           f"{temp_min_heat} and {temp_amb} respectively.")
        assert temp_max_cool > temp_min_cool, (f"temp_max_cool must be greater than temp_min_cool, got {temp_max_cool} "
                                               f"and {temp_min_cool} respectively.")
        assert temp_max_cool <= temp_amb, (f"temp_max_cool must be less than or equal to temp_amb, got {temp_max_cool} "
                                           f"and {temp_amb} respectively.")
        if r_tank is None:
            assert (q_loss is not None) or ((k_ins is not None) and (l_ins is not None) and (h_tank is not None) and
                                            (d_tank is not None)), (f"q_loss must be specified unless r_tank or k_ins, "
                                                                    f" l_ins, h_tank, and d_tank are given.")
            if q_loss is not None:
                if temp_ref_q_loss > temp_amb:
                    assert q_loss >= 0, f"q_loss must be greater than or equal to 0 if defined in heating, got {q_loss}."
                else:
                    assert q_loss <= 0, f"q_loss must be less than or equal to 0 if defined in cooling, got {q_loss}."
            else:
                assert k_ins >= 0, f"k_ins must be greater than or equal to 0, got {k_ins}."
                assert l_ins >= 0, f"l_ins must be greater than or equal to 0, got {l_ins}."
        else:
            assert r_tank >= 0, f"r_tank must be greater than 0, got {r_tank}."

        if c_tank is None:
            assert n_tanks is not None, f"n_tanks must be specified unless c_tank is given."
            assert n_tanks >= 1, f"n_tanks must be greater than or equal to 1, got {n_tanks}."
            if h_tank is not None:
                assert h_tank > 0, f"h_tank must be greater than 0, got {h_tank}."
            if d_tank is not None:
                assert d_tank > 0, f"d_tank must be greater than 0, got {d_tank}."
            assert p_tank > 87, f"p_tank must be greater than atmospheric pressure, got {p_tank}."
            assert (v_tank is not None) or ((h_tank is not None) and (d_tank is not None)), \
                f"v_tank must be specified unless both h_tank and d_tank are given."
        else:
            assert c_tank >= 0, f"c_tank must be greater than 0, got {c_tank}."

        # < Calculate class attributes >
        if r_tank is None:
            # < Use tank parameters to calculate r_tank >
            if (k_ins is not None) and (l_ins is not None):
                # r_tank is characterized by k_ins, l_ins, and tank geometry
                # Thermal resistance to radial conduction through tank walls, Incopera DeWitt Fundementals of Heat
                # Transfer 7th Ed. 3.33
                r2 = d_tank/2
                r1 = r2 - l_ins
                r_tank_horiz = np.log(r2/r1)/(2*np.pi*h_tank*k_ins)*1000
                # Thermal resistance to conduction through the top and bottom tank surfaces
                r_tank_vert = l_ins/k_ins/(2*np.pi*(d_tank**2/4))*1000
                # Resistances in parallel
                r_tank = ((1/r_tank_horiz) + (1/r_tank_vert))**(-1)
            else:
                # r_tank is characterized by q_loss
                r_tank = (temp_ref_q_loss - temp_amb)/q_loss

        if c_tank is None:
            # < Use tank parameters to calculate c_tank >
            # Volumetric capacity of all tanks in m^3
            v_tanks = np.pi*(d_tank**2/4)*h_tank*n_tanks if v_tank is None else v_tank*n_tanks
            # Density of the fluid in kg/m^3, based on average operating temp.and p_tank
            rho_fluid = PropsSI("D", "T", (temp_max_heat+temp_min_cool)/2 + 273.15, "P", p_tank*1000, fluid)
            # Specific heat of the fluid in kJ/kg-K
            cp_fluid = PropsSI("C", "T", (temp_max_heat+temp_min_cool)/2 + 273.15, "P", p_tank*1000, fluid)/1000
            c_tank = rho_fluid*v_tanks*cp_fluid

        # Time constant in h
        tau = r_tank*c_tank/3600
        a = np.exp(-dt/tau)
        b = (1 - a)*tau

        # Stored energy is defined with respect to datum (zero energy point) at temp_amb. Positive x represents heating
        # potential and negative x represents cooling potential
        x_max = c_tank*(temp_max_heat - temp_amb)/3600
        x_min = c_tank*(temp_min_cool - temp_amb)/3600

        # Fit qd_max and qd_min linear models if scalar values were given
        x_inter_qd_max = c_tank*(temp_min_heat - temp_amb)/3600
        x_inter_qd_min = c_tank*(temp_max_cool - temp_amb)/3600
        qd_max_rn = self.__qd_linfit(qd_max_rn, x_max, x_inter_qd_max)
        qd_min_rn = self.__qd_linfit(qd_min_rn, x_min, x_inter_qd_min)
        qd_max_dc = self.__qd_linfit(qd_max_dc, x_max, x_inter_qd_max)
        qd_min_dc = self.__qd_linfit(qd_min_dc, x_min, x_inter_qd_min)

        # Validate qd_max and qd_min linear models
        qd_max_in = (x_max + x_inter_qd_max)/2
        qd_min_in = (x_min + x_inter_qd_min)/2
        qd_max_rn_out = np.polyval(qd_max_rn, qd_max_in)
        qd_min_rn_out = np.polyval(qd_min_rn, qd_min_in)
        qd_max_dc_out = np.polyval(qd_max_dc, qd_max_in)
        qd_min_dc_out = np.polyval(qd_min_dc, qd_min_in)
        assert qd_max_rn_out >= 0, (f"qd_max_rn must be defined as nonnegative when the tank has heating potential, "
                                    f"got {qd_max_rn_out} with x = {qd_max_in}.")
        assert qd_min_rn_out <= 0, (f"qd_min_rn must be defined as nonpositive when the tank has cooling potential, "
                                    f"got {qd_min_rn_out} with x = {qd_min_in}.")
        assert qd_max_dc_out >= 0, (f"qd_max_dc must be defined as nonnegative when the tank has heating potential, "
                                    f"got {qd_max_dc_out} with x = {qd_max_in}.")
        assert qd_min_dc_out <= 0, (f"qd_min_dc must be defined as nonpositive when the tank has cooling potential, "
                                    f"got {qd_min_dc_out} with x = {qd_min_in}.")
        assert not ((qd_max_rn_out == 0) and (qd_min_rn_out >= 0) and (qd_max_dc_out >= 0) and (qd_min_dc_out >= 0)), \
            (f"qd_max_rn, qd_min_rn, qd_max_dc, qd_min_dc are all equal to zero, please specify nonzero "
             f"values for one or more of these variables to enable tank discharging.")

        # < Assign class attributes >
        self.a = a
        self.b = b
        self.x_max = x_max
        self.x_min = x_min
        self.pd_max_rn = pd_max_rn
        self.pd_max_dc = pd_max_dc
        self.qd_max_rn = qd_max_rn
        self.qd_min_rn = qd_min_rn
        self.qd_max_dc = qd_max_dc
        self.qd_min_dc = qd_min_dc


class HeatPump:
    """A class representing a polynomial heat pump model with auxiliary heat.

    Attributes:
        cop_map_heat (np.ndarray): 1-D array containing the coefficients of a polynomial model for heating COP as a
            function of source temperature in degC
        cop_map_cool (np.ndarray): 1-D array containing the coefficients of a polynomial model for cooling COP as a
            function of source temperature in degC
        cop_map_err_max (float): Maximum +- percent error expected in COP map
        cop_map_err_dist (Literal[str]): Distribution of COP error, "normal" or "uniform"
        p_hp_max_heat (np.ndarray): 1-D array containing the coefficients of a polynomial model for heat pump
            electrical power in kW as a function of source temperature in degC for heating mode
        p_hp_max_cool (np.ndarray): 1-D array containing the coefficients of a polynomial model for maximum heat pump
            electrical power in kW as a function of source temperature in degC for cooling mode
        p_hp_err_max (float): Maximum +- percent error expected in power consumption map
        p_hp_err_dist (Literal[str]): Distribution of power consumption error, "normal" or "uniform"
        smooth_win (pd.Timedelta): A timedelta representing the smoothing window for the real metrics which is used to
            better emulate modeling errors
        p_aux_max (float): Maximum electrical power of auxiliary heat in kW
        q_hp_max_heat (np.ndarray, None): 1-D array containing the coefficients of a polynomial model for heat pump
            thermal capacity in kW as a function of source temperature in degC for heating mode, can be omitted
        q_hp_max_cool (np.ndarray, None): 1-D array containing the coefficients of a polynomial model for heat pump
            thermal capacity in kW as a function of source temperature in degC for cooling mode, can be omitted
    """

    def __init__(
           self,
           cop_map_heat: float | list[float] | np.ndarray,
           cop_map_cool: float | list[float] | np.ndarray,
           cop_map_err_max: float | None = None,
           cop_map_err_dist: Literal["normal", "uniform"] | None = None,
           p_hp_max: float | list[float] | np.ndarray | None = None,
           p_hp_max_heat: float | list[float] | np.ndarray | None = None,
           p_hp_max_cool: float | list[float] | np.ndarray | None = None,
           p_hp_err_max: float | None = None,
           p_hp_err_dist: Literal["normal", "uniform"] | None = None,
           smooth_win: float | pd.Timedelta | None = None,
           p_aux_max: float | None = None,
           q_hp_max: float | list[float] | np.ndarray | None = None,
           q_hp_max_heat: float | list[float] | np.ndarray | None = None,
           q_hp_max_cool: float | list[float] | np.ndarray | None = None):
        """Initialize attributes of the HeatPump class.

        Parameters:
            cop_map_heat: 1-D array containing the coefficients of a polynomial model for heating COP as a function of
                source temperature in degC
            cop_map_cool: 1-D array containing the coefficients of a polynomial model for cooling COP as a function of
                source temperature in degC
            cop_map_err_max: Maximum +- percent error expected in COP map
            cop_map_err_dist: Distribution of COP error, "normal" or "uniform"
            p_hp_max: 1-D array containing the coefficients of a polynomial model for heat pump
                electrical power as a function of source temperature in degC (assuming it is the same for both heating
                and cooling mode)
            p_hp_max_heat: 1-D array containing the coefficients of a polynomial model for heat pump electrical power as
                a function of source temperature in degC for heating mode
            p_hp_max_cool: 1-D array containing the coefficients of a polynomial model for heat pump electrical power as
                a function of source temperature in degC for cooling mode
            p_hp_err_max: Maximum +- percent error expected in power consumption map
            p_hp_err_dist: Distribution of power consumption error, "normal" or "uniform"
            smooth_win: A time in hours or timedelta representing the smoothing window for the real metrics which is
                used to better emulate modeling errors
            p_aux_max: Maximum electrical power of auxiliary heat in kW
            q_hp_max: 1-D array containing the coefficients of a polynomial model for heat pump thermal capacity
                in kW as a function of source temperature in degC (assuming it is the same for both heating
                and cooling mode)
            q_hp_max_heat: 1-D array containing the coefficients of a polynomial model for heat pump thermal capacity
                in kW as a function of source temperature in degC for heating mode
            q_hp_max_cool: 1-D array containing the coefficients of a polynomial model for heat pump thermal capacity
                in kW as a function of source temperature in degC for cooling mode
        """
        # < Set default params >
        if cop_map_err_max is None:
            cop_map_err_max = 0.0
        if p_hp_err_max is None:
            p_hp_err_max = 0.0
        if cop_map_err_dist is None:
            if p_hp_err_dist is not None:
                cop_map_err_dist = p_hp_err_dist
            else:
                cop_map_err_dist = "normal"
        if p_hp_err_dist is None:
            p_hp_err_dist = cop_map_err_dist
        if p_hp_max_heat is None:
            p_hp_max_heat = p_hp_max
        if p_hp_max_cool is None:
            p_hp_max_cool = p_hp_max
        if smooth_win is None:
            smooth_win = 0.0
        if p_aux_max is None:
            p_aux_max = 0.0
        if q_hp_max is not None:
            if q_hp_max_heat is None:
                q_hp_max_heat = q_hp_max
            if q_hp_max_cool is None:
                q_hp_max_cool = q_hp_max

        # < Condition params >
        cop_map_heat = make_1d(cop_map_heat)
        cop_map_cool = make_1d(cop_map_cool)
        cop_map_err_dist = cop_map_err_dist.lower()
        p_hp_err_dist = p_hp_err_dist.lower()
        p_hp_max_heat = make_1d(p_hp_max_heat)
        p_hp_max_cool = make_1d(p_hp_max_cool)
        smooth_win = set_timedelta(smooth_win)
        if q_hp_max_heat is not None:
            q_hp_max_heat = make_1d(q_hp_max_heat)
        if q_hp_max_cool is not None:
            q_hp_max_cool = make_1d(q_hp_max_cool)

        # < Validate params >
        assert np.polyval(cop_map_heat, 20.0) > 0, (f"cop_map_heat should return a positive value with a reasonable "
                                                    f"source temperature, got {float(np.polyval(cop_map_heat, 20.0))}"
                                                    f" with temp_source = 20.")
        assert cop_map_err_max > 0, f"cop_map_err_max must be greater than 0, got {cop_map_err_max}."
        assert (cop_map_err_dist == "normal") or (cop_map_err_dist == "uniform"), \
            f"cop_map_err_dist must be \"normal\" or \"uniform\" got {cop_map_err_dist}."
        assert p_hp_max_heat is not None, f"p_hp_max_heat must be specified unless p_hp_max is given."
        assert p_hp_max_cool is not None, f"p_hp_max_cool must be specified unless p_hp_max is given."
        assert np.polyval(p_hp_max_heat, 20.0) > 0, (f"p_hp_max_heat should return a positive value with a reasonable "
                                                     f"source temperature, got {float(np.polyval(p_hp_max_heat, 20.0))}"
                                                     f" with temp_source = 20.")
        assert np.polyval(p_hp_max_cool, 10.0) > 0, (f"p_hp_max_cool should return a positive value with a reasonable "
                                                     f"source temperature, got {float(np.polyval(p_hp_max_cool, 10.0))}"
                                                     f" with temp_source = 10.")
        assert smooth_win >= pd.Timedelta(0), f"smooth_win must be nonnegative, got {smooth_win}."
        assert p_aux_max >= 0, f"p_aux_max must be greater than or equal to 0, got {p_aux_max}."
        if q_hp_max_heat is not None:
            assert np.polyval(q_hp_max_heat, 20.0) > 0, (f"q_hp_max_heat should return a positive value with a reasonable "
                                                         f"source temperature, got {float(np.polyval(q_hp_max_heat, 20.0))}"
                                                         f" with temp_source = 20.")

        # < Assign class attributes >
        self.cop_map_heat = cop_map_heat
        # Ensure efficiency convention is followed
        if np.polyval(cop_map_cool, 10.0) > 0:
            self.cop_map_cool = -cop_map_cool
        else:
            self.cop_map_cool = cop_map_cool
        self.cop_map_err_max = cop_map_err_max
        self.cop_map_err_dist = cop_map_err_dist
        self.p_hp_max_heat = p_hp_max_heat
        self.p_hp_max_cool = p_hp_max_cool
        self.p_hp_err_max = p_hp_err_max
        self.p_hp_err_dist = p_hp_err_dist
        self.smooth_win = smooth_win
        self.p_aux_max = p_aux_max
        self.q_hp_max_heat = q_hp_max_heat
        # Ensure heat transfer convention is followed
        if (q_hp_max_cool is not None) and (np.polyval(q_hp_max_cool, 10.0) > 0):
            self.q_hp_max_cool = -q_hp_max_cool
        else:
            self.q_hp_max_cool = q_hp_max_cool

    @staticmethod
    def __gen_errfact(
            max_err: float,
            dist: Literal["normal", "uniform"],
            inpt_arr: np.ndarray,
            rng: np.random.Generator | None = None) -> np.ndarray:
        """Generates an array of error factors of based on the maximum error bounds, error distribution, and size of
        input array.

        Parameters:
            max_err: Maximum +- percentage error bound
            dist: Error distribution, "normal" or "uniform"
            inpt_arr: Array of the desired size and shape for err_fact
            rng: An instance of the np.random.Generator class

        Output:
            err_fact: Error factor array
        """
        if rng is None:
            rng = np.random.default_rng()
        size = inpt_arr.size
        shape = inpt_arr.shape
        if dist == "normal":
            err_fact = 1 + rng.normal(0, max_err/200, shape)
        else:
            err_fact = 1 + (rng.random(size)*2 - 1)*max_err/100
            err_fact = np.reshape(err_fact, shape)
        return err_fact

    def calc_heat_pump_performance(
            self,
            metric: Literal["cop", "p_hp_max", "q_hp_max"],
            temp_source_real: float | list[float] | np.ndarray,
            temp_source_pred: float | list[float] | np.ndarray | None = None,
            dt: float | list[float] | np.ndarray | None = None,
            rng: np.random.Generator | None = None) -> dict:
        """Calculates the resulting real and predicted heating and cooling COP or p_hp_max arrays based on the inputted
        source temperature arrays, then packages them in a dict.

        Parameters:
            metric: The performance metric, "cop" or "p_hp_max" to be calculated
            temp_source_real: 1-D array of the real source temperature in degC
            temp_source_pred: 2-D array of the predicted source temperature in degC
            dt: The simulation timestep in h or as timedelta, required when self.smooth_win is set to nonzero value
            rng: An instance of the np.random.Generator class

        Output:
            perf: The resulting performance metric dict
        """

        # Condition params
        if temp_source_pred is None:
            temp_source_pred = temp_source_real
        assert metric == "cop" or metric == "p_hp_max" or metric == "q_hp_max", \
            f"metric must be \"cop\", \"p_hp_max\"or \"q_hp_max\"."
        assert isinstance(temp_source_real, float | int | list | np.ndarray), (f"temp_source_real must be float, int, "
                                                                               f"or array_like.")
        assert isinstance(temp_source_pred, float | int | list | np.ndarray), (f"temp_source_pred must be float, int, "
                                                                               f"or array_like.")

        # Validate params
        temp_source_real = make_array(temp_source_real)
        temp_source_pred = make_array(temp_source_pred)
        dt = set_timedelta(dt)

        # Choose parameters based on metric
        if metric == 'cop':
            err_max = self.cop_map_err_max
            err_dist = self.cop_map_err_dist
            map_heat = self.cop_map_heat
            map_cool = self.cop_map_cool
        elif metric == "p_hp_max":
            err_max = self.p_hp_err_max
            err_dist = self.p_hp_err_dist
            map_heat = self.p_hp_max_heat
            map_cool = self.p_hp_max_cool
        else:
            # Do not add error
            err_max = 0
            err_dist = None
            if self.q_hp_max_heat is None:
                # Prepare to fill array with inf
                map_heat = np.full(1, np.inf)
            else:
                map_heat = self.q_hp_max_heat
            if self.q_hp_max_cool is None:
                # Prepare to fill array with -inf
                map_cool = np.full(1, -np.inf)
            else:
                map_cool = self.q_hp_max_cool

        # Generate scaling factors to account for COP or p_hp map error under known source conditions
        heat_errfact = self.__gen_errfact(err_max, err_dist, temp_source_real, rng)
        cool_errfact = self.__gen_errfact(err_max, err_dist, temp_source_real, rng)

        # Calculate performance metric vectors
        metric_heat_real = np.multiply(np.polyval(map_heat, temp_source_real), heat_errfact)
        metric_cool_real = np.multiply(np.polyval(map_cool, temp_source_real), cool_errfact)
        metric_heat_pred = np.polyval(map_heat, temp_source_pred)
        metric_cool_pred = np.polyval(map_cool, temp_source_pred)

        # Apply smoothing to real metrics if necessary
        if self.smooth_win >= dt/2:
            metric_heat_real = smoothdata(metric_heat_real, self.smooth_win, dt)
            metric_cool_real = smoothdata(metric_cool_real, self.smooth_win, dt)
        # Store in dict and return
        return {metric + '_heat_real': metric_heat_real, metric + '_cool_real': metric_cool_real,
                metric + '_heat_pred': metric_heat_pred, metric + '_cool_pred': metric_cool_pred}


class Load:
    """A class representing a polynomial thermal load model.

    Attributes:
        load_line_heat (np.ndarray): 1-D array containing the coefficients of a polynomial heating load line model
            for thermal load in kW from outdoor temperature in degC
        load_line_cool (np.ndarray): 1-D array containing the coefficients of a polynomial cooling load line model
            for thermal load in kW from outdoor temperature in degC
        t_movmean (pd.Timedelta): The window length for the moving average applied to outdoor temperature data
            before it is inputted to the load line models
    """

    def __init__(
            self,
            load_line_heat: list[float] | np.ndarray,
            load_line_cool: list[float] | np.ndarray,
            t_movmean: float | pd.Timedelta | None = None):
        """Initialize attributes of the Load class.

        Parameters:
        load_line_heat: 1-D array containing the coefficients of a polynomial heating load line model
            for thermal load in kW from outdoor temperature in degC
        load_line_cool: 1-D array containing the coefficients of a polynomial cooling load line model
            for thermal load in kW from outdoor temperature in degC
        t_movmean: The window length for the moving average applied to outdoor temperature data
            before it is inputted to the load line models
        """

        # < Condition params >
        load_line_heat = make_1d(load_line_heat)
        load_line_cool = make_1d(load_line_cool)
        t_movmean = set_timedelta(t_movmean)

        # < Validate params >
        assert np.polyval(load_line_heat, -5.0) > 0, (
            f"load_line_heat should return a positive value with a reasonable "
            f"source temperature, got {float(np.polyval(load_line_heat, -5.0))}"
            f" with theta = -5.0.")
        assert np.polyval(load_line_cool, 25.0) < 0, (
            f"load_line_cool should return a negative value with a reasonable "
            f"source temperature, got {float(np.polyval(load_line_cool, 25.0))}"
            f" with theta = 25.0.")

        # < Assign class attributes >
        self.load_line_heat = load_line_heat
        self.load_line_cool = load_line_cool
        self.t_movmean = t_movmean

    def calc_load_pred(self, theta_pred: np.ndarray, dt: float | pd.Timedelta | None = None) -> np.ndarray:
        """Calculates the predicted thermal load in kW from the predicted outdoor temperature array in degC.

        Parameters:
            theta_pred: A 2-D array of the simulation forecasted outdoor temperature in degC
            dt: The theta_pred timestep/frequency in h or timedelta

        Output:
            load_pred: The resulting thermal load array in kW
        """
        dt = set_timedelta(dt)
        if self.t_movmean is not None:
            assert (dt is not None), f"dt must be given if t_movmean is given."
            theta_pred_movmean = mov_mean(theta_pred, round(self.t_movmean/dt))
        else:
            theta_pred_movmean = theta_pred

        # < Calculate load >
        load_pred = (np.maximum(0, np.polyval(self.load_line_heat, theta_pred_movmean)) +
                     np.minimum(0, np.polyval(self.load_line_cool, theta_pred_movmean)))
        return load_pred
