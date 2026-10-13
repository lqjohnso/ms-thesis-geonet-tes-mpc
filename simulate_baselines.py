"""
========================================================================================================================
SCRIPT NAME:    simulate_baselines.py
PROJECT:        CHPB-75
AUTHOR(S):      Liam Johnson
AFFILIATION:    Purdue University, CHPB Herrick Labs
DATE CREATED:   8-20-2026
LAST MODIFIED:  8-30-2026
VERSION:        1.2
PYTHON VERSION: 3.14.6
========================================================================================================================
DESCRIPTION: This module contains the functions which run simulations for baseline control.

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

from dynamics import Control
from utils import clamp
import numpy as np
from dynamics import State
from scenario import Scenario

######################
# Baseline Functions #
######################


def simulate_no_tes_baseline(s: Scenario):
    """Simulates the no TES baseline case.

    Parameters:
        m: An instance of the Models class.
        s: An instance of the Scenario class for the optimization horizon which has been reduced with Scenario.reduce()

    Outputs:
        state: An instance of the State class with the recorded tank state
        control: An instance of the Control class with the recorded control inputs
    """

    # < Setup >
    # Heating load indices for ReNEWW House
    h_rn = np.where(s.rn.load > 0)[0]
    # Cooling load indices for ReNEWW House
    c_rn = np.where(s.rn.load <= 0)[0]
    # Heating load indices for DC House
    h_dc = np.where(s.dc.load > 0)[0]
    # Cooling load indices for DC House
    c_dc = np.where(s.dc.load <= 0)[0]

    # < Create blank Control and state instances filled with zeros >
    control = Control.blank(s, fill_val=0.0)
    state = State.blank(s, fill_val=0.0)

    # < Calculate control attributes under no TES conditions and assign >
    # Heat pump electrical power
    control.p_gshp_rn[h_rn] = clamp(np.divide(s.rn.load[h_rn], s.rn.cop_heat[h_rn]), s.rn.p_hp_max_heat[h_rn])
    control.p_gshp_rn[c_rn] = clamp(np.divide(s.rn.load[c_rn], s.rn.cop_cool[c_rn]), s.rn.p_hp_max_cool[c_rn])
    control.p_ashp[h_dc] = clamp(np.divide(s.dc.load[h_dc], s.dc.cop_heat[h_dc]), s.dc.p_hp_max_heat[h_dc])
    control.p_ashp[c_dc] = clamp(np.divide(s.dc.load[c_dc], s.dc.cop_cool[c_dc]), s.dc.p_hp_max_cool[c_dc])

    # Heat pump thermal power
    control.q_gshp_rn[h_rn] = np.multiply(control.p_gshp_rn[h_rn], s.rn.cop_heat[h_rn])
    control.q_gshp_rn[c_rn] = np.multiply(control.p_gshp_rn[c_rn], s.rn.cop_cool[c_rn])
    control.q_ashp[h_dc] = np.multiply(control.p_ashp[h_dc], s.dc.cop_heat[h_dc])
    control.q_ashp[c_dc] = np.multiply(control.p_ashp[c_dc], s.dc.cop_cool[c_dc])

    # Aux heat
    control.p_aux_gshp[h_rn] = clamp(s.rn.load[h_rn] - control.q_gshp_rn[h_rn], s.rn.p_aux_max[h_rn], 0)
    control.p_aux_ashp[h_dc] = clamp(s.dc.load[h_dc] - control.q_ashp[h_dc], s.dc.p_aux_max[h_dc], 0)

    # Total thermal power
    control.q_tot_del_rn = control.qd_rn + control.q_gshp_rn + control.p_aux_gshp
    control.q_tot_del_dc = control.qd_dc + control.q_ashp + control.p_aux_ashp + control.q_gshp_dc

    # Duty cycle
    control.d_gshp_rn[h_rn] = np.divide(control.p_gshp_rn[h_rn], s.rn.p_hp_max_heat[h_rn])
    control.d_gshp_rn[c_rn] = np.divide(control.p_gshp_rn[c_rn], s.rn.p_hp_max_cool[c_rn])
    control.d_ashp[h_dc] = np.divide(control.p_ashp[h_dc], s.dc.p_hp_max_heat[h_dc])
    control.d_ashp[c_dc] = np.divide(control.p_ashp[c_dc], s.dc.p_hp_max_cool[c_dc])
    control.d_aux_ashp[h_dc] = np.divide(control.p_aux_ashp[h_dc], s.dc.p_aux_max[h_dc])

    return state, control
