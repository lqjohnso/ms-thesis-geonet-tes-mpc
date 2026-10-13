"""
========================================================================================================================
SCRIPT NAME:    __main__.py
PROJECT:        CHPB-75
AUTHOR(S):      Liam Johnson
AFFILIATION:    Purdue University, CHPB Herrick Labs
DATE CREATED:   7-29-2026
LAST MODIFIED:  10-06-2026
VERSION:        1.5
PYTHON VERSION: 3.14.6
========================================================================================================================
DESCRIPTION: This module is used to simulate a Model Predictive Controller (MPC) for a sensible Thermal Energy Storage
(TES) based residential geothermal district heating/cooling system; accounting for modeling and forecast errors. This
code calls all relevant subfunctions which set up and run simulations of sub-optimal, optimal, and rule-based control
under the conditions specified in the config and data files. All code conforms to Python best practices, PEP 8 style
guidelines, and object-oriented programming fundamentals. Requires an internet connection.

DEPENDENCIES:
    - coolprop
    - cvxpy
    - matplotlib
    - numpy
    - pandas
    - paperplot-quantum
    - python-dotenv
    - python-openei
    - requests
    - scipy
    - toml
    - tqdm
========================================================================================================================
"""

####################
# Required Imports #
####################

from build_models import build_models
from build_scenario import build_scenario
from dotenv import load_dotenv
from utils import get_env_var
from metrics import calculate_metrics
from output import generate_output
from pathlib import Path
from simulate import simulate

##############################
# Load Environment Variables #
##############################

try:
    load_dotenv()
except FileNotFoundError:
    print("A .env file must be present in the current directory, please create one using .env.example.")

#################
# Configuration #
#################

# < Set config filenames to use for simulation >
models_config = 'models_config.toml'
scenario_config = 'scenario_config_48h_heat.toml'
output_config = 'output_config.toml'

# < Save absolute filepaths of config files >
curr_dir = Path.cwd()
config_dir = curr_dir / get_env_var("CONFIG_DIR", "config")
config_files = dict()
config_files['models'] = config_dir / models_config
config_files['scenario'] = config_dir / scenario_config
config_files['output'] = config_dir / output_config

#########################
# Build Class Instances #
#########################

print("Building models...")
models = build_models(config_files)
print("Building scenario...")
scenario = build_scenario(models, config_files)

########################
# Simulate Controllers #
########################

results = dict()
for controller in scenario.controllers:
    print("Simulating " + controller.lower() + " controller...")
    results[controller] = simulate(controller, models, scenario)

#####################
# Calculate Metrics #
#####################

results = calculate_metrics(results)

###################
# Generate Output #
###################

print("Generating output...")
generate_output(models, scenario, results, config_files)
