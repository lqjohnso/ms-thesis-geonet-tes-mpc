"""
========================================================================================================================
SCRIPT NAME:    build_models.py
PROJECT:        CHPB-75
AUTHOR(S):      Liam Johnson
AFFILIATION:    Purdue University, CHPB Herrick Labs
DATE CREATED:   7-29-2026
LAST MODIFIED:  8-16-2026
VERSION:        1.2
PYTHON VERSION: 3.14.6
========================================================================================================================
DESCRIPTION: This module contains code which builds the models class which stores instances of tank and heat pump
classes used in the simulation.

DEPENDENCIES:
    - coolprop
    - numpy
    - pandas
    - toml
========================================================================================================================
"""

####################
# Required Imports #
####################

from config_loader import instantiate_class_from_toml
from config_loader import table_exists
from dataclasses import dataclass
from models import HeatPump
from models import Tank
from models import Load

###########################
# Parent Class Definition #
###########################


@dataclass
class Models:
    """A dataclass used exclusively to store instances of tank and heat pump classes.

    Attributes:
        tank_real: An instance of the Tank class representing real tank dynamics
        tank_est: An instance of the Tank class representing estimated tank dynamics
        gshp: An instance of the HeatPump class representing the ReNEWW House ground source heat pump
        ashp: An instance of the HeatPump class representing the DC House air source heat pump
        load_rn: An instance of the Load class representing ReNEWW House thermal load models
        load_dc: An instance of the Load class representing DC House thermal load models
    """
    # < Declare attributes >
    tank_real: Tank
    tank_est: Tank
    gshp: HeatPump
    ashp: HeatPump
    load_rn: Load
    load_dc: Load

#######################
# Class Instantiation #
#######################


def build_models(config_files: dict) -> Models:
    """Builds an instance of the Model class from config files."""
    # < Child classes >
    tank_real = instantiate_class_from_toml(Tank, [config_files['models'], config_files['scenario']],
                                            ['tank_real', None])
    if table_exists(config_files['models'], 'tank_est'):
        tank_est = instantiate_class_from_toml(Tank, [config_files['models'], config_files['scenario']],
                                           ['tank_est', None])
    else:
        # Assume tank model is perfect if tank_est is not provided
        tank_est = tank_real

    gshp = instantiate_class_from_toml(HeatPump, config_files['models'], 'gshp')
    ashp = instantiate_class_from_toml(HeatPump, config_files['models'], 'ashp')
    load_rn = instantiate_class_from_toml(Load, config_files['models'], 'load_rn')
    load_dc = instantiate_class_from_toml(Load, config_files['models'], 'load_dc')

    # < Parent class >
    return Models(tank_real, tank_est, gshp, ashp, load_rn, load_dc)
