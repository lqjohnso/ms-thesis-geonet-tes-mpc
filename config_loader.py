"""
========================================================================================================================
SCRIPT NAME:    config_loader.py
PROJECT:        CHPB-75
AUTHOR(S):      Liam Johnson
AFFILIATION:    Purdue University, CHPB Herrick Labs
DATE CREATED:   7-29-2026
LAST MODIFIED:  10-04-2026
VERSION:        1.4
PYTHON VERSION: 3.14.6
========================================================================================================================
DESCRIPTION: This module contains all functions used to load config files.

DEPENDENCIES:
    - toml
========================================================================================================================
"""

####################
# Required Imports #
####################

from pathlib import Path
import shutil
import toml
from typing import TypeVar

#############
# Functions #
#############

T = TypeVar("T")


def instantiate_class_from_dict(cls: T, dct: dict) -> T:
    """Instantiates a class from a dictionary given key values match those in __init__ of said class."""
    # Remove all keys that are not used to instantiate class
    filt_dct = {k: v for k, v in dct.items() if k in cls.__init__.__code__.co_varnames}
    # Instantiate class from filtered dict
    return cls(**filt_dct)


def load_toml_config(path: str | Path, table: str | None = None) -> dict:
    """Reads a .toml config to dict from the given path; can read from a specific
    table if specified."""

    # < Validate params >
    path = Path(path)
    if path.is_absolute():
        add_msg = ""
    else:
        add_msg = (" , the filepath given is not absolute, which likely caused the path to be relative to incorrect "
                   "directory")
    assert path.exists(), f"filepath does not exist: {path}{add_msg}."

    # < Load config as dictionary >
    with open(path, "r") as f:
        dct = toml.load(f)
    if table is not None:
        try:
            return dct[table]
        except KeyError:
            print(f"table {table} not found in {path}.")
    else:
        return dct


def table_exists(path: str | Path, table: str | None = None):
    """Checks if the table exists in the .toml config file at path."""
    dct = load_toml_config(path)
    return table in dct


def instantiate_class_from_toml(cls: object, paths: str | Path | list, tables: str | list | None = None,
                                add_attr: dict | None = None) -> object:
    """Instantiates a class directly from .toml config files at paths; can read from specific tables if specified.

    Parameters:
        cls: The class to instantiate
        paths: Single filepath or list of filepaths names to read configs from
        tables: Single table or list of tables to read from in config, note that the length of the tables list
            must be the same as that of filepaths if it is specified, for example,
            instantiate_class_from_toml(cls, [path1, path1, path2], [table1, table2, None]) will read from two
            different tables in config1 and read directly from config2
        add_attr: A dict containing additional attributes under keynames matching those in __init__ of cls

    Output:
        cls_inst: The instantiated class
    """
    if isinstance(paths, list):
        # < Create dict with key value pairs from each filepath >
        dct = {}
        if tables is not None:
            assert len(paths) == len(tables), ("paths and tables lists must be the same length, specify a "
                                               "table for each path listed or use None as a placeholder.")
            for i, path in enumerate(paths):
                dct = dct | load_toml_config(path, tables[i])
        else:
            for path in paths:
                dct = dct | load_toml_config(path)
    else:
        assert not isinstance(tables, list), ("paths and tables lists must be the same length, specify a path for "
                                              "each table listed.")
        # < Create dict with key value pairs in paths >
        dct = load_toml_config(paths, tables)

    # < Add additional attributes to dict >
    if add_attr is None:
        add_attr = dict()
    dct = dct | add_attr

    return instantiate_class_from_dict(cls, dct)


def copy_config(start_path: str | Path, dest_dir: str | Path, ):
    """Copies the config file at start_path to the directory specified by dest_dir. """
    # Get from .env but fall back to default if not present, define with respect to path of calling module
    start_path = Path(start_path)
    dest_path = Path(dest_dir) / start_path.name
    shutil.copy(start_path, dest_path)

