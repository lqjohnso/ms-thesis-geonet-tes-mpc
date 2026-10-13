"""
========================================================================================================================
SCRIPT NAME:    metrics.py
PROJECT:        CHPB-75
AUTHOR(S):      Liam Johnson
AFFILIATION:    Purdue University, CHPB Herrick Labs
DATE CREATED:   8-20-2026
LAST MODIFIED:  10-07-2026
VERSION:        1.2
PYTHON VERSION: 3.14.6
========================================================================================================================
DESCRIPTION: This module contains the function which calculates cost savings and maxmimum peak demand metrics based on
the results of each controller simulation.

DEPENDENCIES:
    - numpy
    - pandas
========================================================================================================================
"""

####################
# Required Imports #
####################

import numpy as np
from utils import calc_abs_savings
from utils import calc_perc_savings

####################
# Metrics Function #
####################


def calculate_metrics(results: dict) -> dict:
    """Calculates cost savings metrics based on the results of each controller simulation."""
    # < Get baseline costs if available >
    if "No TES Baseline" in results:
        elec_cost_no_tes = results["No TES Baseline"]['elec_cost_obj']
        peak_demand_rn_no_tes = results["No TES Baseline"]['peak_demand_rn']
        peak_demand_dc_no_tes = results["No TES Baseline"]['peak_demand_dc']
        peak_demand_tot_no_tes = results["No TES Baseline"]['peak_demand_tot']
    else:
        elec_cost_no_tes = np.nan
        peak_demand_rn_no_tes = np.nan
        peak_demand_dc_no_tes = np.nan
        peak_demand_tot_no_tes = np.nan
    if "RBC TES Baseline" in results:
        elec_cost_tes = results["RBC TES Baseline"]['elec_cost_obj']
        peak_demand_rn_tes = results["RBC TES Baseline"]['peak_demand_rn']
        peak_demand_dc_tes = results["RBC TES Baseline"]['peak_demand_dc']
        peak_demand_tot_tes = results["RBC TES Baseline"]['peak_demand_tot']
    else:
        elec_cost_tes = np.nan
        peak_demand_rn_tes = np.nan
        peak_demand_dc_tes = np.nan
        peak_demand_tot_tes = np.nan

    for controller in results.keys():
        # < Calculate cost savings metrics >
        results[controller]['cost_savings_baseline_no_tes'] = (
            calc_abs_savings(results[controller]['elec_cost_obj'], elec_cost_no_tes))
        results[controller]['cost_savings_baseline_tes'] = (
            calc_abs_savings(results[controller]['elec_cost_obj'], elec_cost_tes))
        results[controller]['perc_cost_savings_baseline_no_tes'] = (
            calc_perc_savings(results[controller]['elec_cost_obj'], elec_cost_no_tes))
        results[controller]['perc_cost_savings_baseline_tes'] = (
            calc_perc_savings(results[controller]['elec_cost_obj'], elec_cost_tes))
        results[controller]['perc_peak_red_rn_baseline_no_tes'] = (
            calc_perc_savings(results[controller]['peak_demand_rn'], peak_demand_rn_no_tes))
        results[controller]['perc_peak_red_dc_baseline_no_tes'] = (
            calc_perc_savings(results[controller]['peak_demand_dc'], peak_demand_dc_no_tes))
        results[controller]['perc_peak_red_tot_baseline_no_tes'] = (
            calc_perc_savings(results[controller]['peak_demand_tot'], peak_demand_tot_no_tes))
        results[controller]['perc_peak_red_rn_baseline_tes'] = (
            calc_perc_savings(results[controller]['peak_demand_rn'], peak_demand_rn_tes))
        results[controller]['perc_peak_red_dc_baseline_tes'] = (
            calc_perc_savings(results[controller]['peak_demand_dc'], peak_demand_dc_tes))
        results[controller]['perc_peak_red_tot_baseline_tes'] = (
            calc_perc_savings(results[controller]['peak_demand_tot'], peak_demand_tot_tes))

    return results
