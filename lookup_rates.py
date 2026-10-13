"""
========================================================================================================================
SCRIPT NAME:    lookup_rates.py
PROJECT:        CHPB-75
AUTHOR(S):      firstof9, Liam Johnson
AFFILIATION:    Purdue University, CHPB Herrick Labs
DATE CREATED:   7-31-2026
LAST MODIFIED:  8-27-2026
VERSION:        1.2
PYTHON VERSION: 3.14.6
========================================================================================================================
DESCRIPTION: This module contains code which allows the user to search the OpenEI database for electrical rate plans by
using lat&lon coordinates and a search radius. Based on code from python-openei documentation:
https://pypi.org/project/python-openei/

DEPENDENCIES:
    - python-dotenv
    - python-openei
========================================================================================================================
"""

####################
# Required Imports #
####################

import asyncio
from dotenv import load_dotenv
from openeihttp import Rates
from os import getenv

##############################
# Load Environment Variables #
##############################

try:
    load_dotenv()
except FileNotFoundError:
    print("A .env file must be present in the current directory, please create one using .env.example.")

###################
# Search Criteria #
###################

# Set to None if not used
# Address as in Google Maps
address = '541 Hayes St, West Lafayette, IN 47906, United States'
# Or
# Coordinates as list
coords = None
# Search radius in miles, use 0 to pinpoint location
radius = 0

###################
# Lookup Function #
###################


async def lookup_rates(address: str | None = None, crds: list | None = None, r: float = 0):
    """Prints information about utility plans in the OpenEI database for an area defined by coordinates and a search
    radius.

    Parameters:
        crds: A 2-D list of the latitude and longitude coordinates
        r: The search radius in miles with the central point as the given coordinates
    """
    assert (address is not None) or (crds is not None), f"Either address or crds must be specified"
    if crds is not None:
        assert isinstance(crds, list) and len(crds) == 2, f"coords must be a list of length 2."
    assert isinstance(r, float | int), f"radius must be float or int."

    api_key = None
    msg = f"Please set OPENEI_API_KEY in the .env file as in .env.example."
    try:
        api_key = getenv("OPENEI_API_KEY")
    except KeyError:
        print(msg)
    assert (api_key is not None) and len(api_key) > 0, msg

    # < Set up lookup using coordinates and search radius >
    if coords is not None:
        api = Rates(
            api=api_key,
            lat=crds[0],
            lon=crds[1],
            radius=r
        )
    else:
        api = Rates(
            api=api_key,
            address=address,
            radius=r
        )

    plans = await api.lookup_plans()

    # Plans will be grouped by utility name
    for utility, plan_list in plans.items():
        if utility != "Not Listed":
            print(f"\nUtility: {utility}")
            for plan in plan_list:
                print(f"  - {plan['name']} (Plan Label: {plan['label']}, "
                      f"Link: https://apps.openei.org/USURDB/rate/view/{plan['label']})")

#######################
# Run Lookup Function #
#######################

asyncio.run(lookup_rates(address, coords, radius))
