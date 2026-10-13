"""
========================================================================================================================
SCRIPT NAME:    build_scenario.py
PROJECT:        CHPB-75
AUTHOR(S):      Liam Johnson
AFFILIATION:    Purdue University, CHPB Herrick Labs
DATE CREATED:   8-02-2026
LAST MODIFIED:  8-13-2026
VERSION:        1.1
PYTHON VERSION: 3.14.6
========================================================================================================================
DESCRIPTION: This module contains code which builds the scenario class that stores attributes relating to
general and house-specific simulation conditions.

DEPENDENCIES:
    - coolprop
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

from build_models import Models
from config_loader import instantiate_class_from_toml
from scenario import Scenario

#######################
# Class Instantiation #
#######################


def build_scenario(models: Models, config_files: dict) -> Scenario:
    """Builds an instance of the Scenario class from the Models instance and config files."""
    # < Scenario Instance >
    scenario = instantiate_class_from_toml(Scenario, [config_files['scenario'], config_files['models']],
                                           [None, 'gshp'], {'models': models})
    return scenario
