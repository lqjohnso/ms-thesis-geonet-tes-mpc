"""
========================================================================================================================
SCRIPT NAME:    output.py
PROJECT:        CHPB-75
AUTHOR(S):      Liam Johnson
AFFILIATION:    Purdue University, CHPB Herrick Labs
DATE CREATED:   8-18-2026
LAST MODIFIED:  10-08-2026
VERSION:        1.5
PYTHON VERSION: 3.14.6
========================================================================================================================
DESCRIPTION: This module contains all classes and functions which create plots, print to terminal, and save results of
simulations.

DEPENDENCIES:
    - coolprop
    - matplotlib
    - numpy
    - pandas
    - paperplot-quantum
    - python-openei
    - requests
    - scipy
    - toml
========================================================================================================================
"""

####################
# Required Imports #
####################

from config_loader import copy_config
from config_loader import instantiate_class_from_toml
from copy import deepcopy
import datetime as dtm
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import inspect
import json
import numpy as np
import pandas as pd
import paperplot as pp
from pathlib import Path
from typing import Literal
from utils import forecast_arr_to_vec
from utils import pos_dict

################
# Output Class #
################


class Output:
    """A class containing information about plotting and data storage.

    Attributes:
        style (str): The style used for plotting with paperplot
        heat_color (str): The color used to represent heating in plots
        cool_color (str): The color used to represent cooling in plots
        energy_rate_color (str): The color used to represent energy rates in plots
        demand_rate_color (str): The color used to represent demand rates in plots
        soc_color (str): The color used to represent tank state of charge in plots
        charge_color (str): The color used to represent TES charging in plots
        disch_color_rn (str): The color used to represent ReNEWW House discharging in plots
        disch_color_dc (str): The color used to represent DC House discharging in plots
        ashp_color (str): The color used to represent ASHP in plots
        gshp_rn_color (str): The color used to represent GSHP to ReNEWW in plots
        gshp_dc_color (str): The color used to represent GSHP to DC in plots
        ashp_aux_color (str): The color used to represent ASHP aux heat in plots
        gshp_aux_color (str): The color used to represent GSHP aux heat in plots
        tot_color (str): The color used to represent total quantities in plots
        load_color (str): The color used to represent house thermal load in plots
        sec_linestyle (str): The linestyle used to represent predicted or other secondary quantities in plots
        flows_style ("plot", "stacked_area"): The style of plot used to represent heat, electrical, and duty cycle
        font_scale (float): The scaling factor applied to the font size in style
        linewidth (float, None): The figure linewidth
        aspect_ratio (float, None): The aspect ratio for plots (height/width), defaults to golden ratio
        fig_width (str): The figure width in columns, 'single', 'double'
        lgd_loc (str): Setting for ax.legend(loc=_)
        lgd_col (int, str): The number of columns in the legend, can be set to 'max' for a horizontal legend
        lgd_bbox (tuple): Setting for ax.legend(bbox_to_anchor=_)
        lgd_frame (bool): Whether the legend box is visible
        lgd_hndl_len (int): Setting for ax.legend(handlelength=_)
        grid (bool, None): Whether to include the grid on plots
        pts (int): The number of points used for plotting when applicable
        date_format (str): Date format used for timeseries plots
        all_pos (bool): Whether to take the absolute value of all results for plotting
        as_subplots (bool): Whether to group like plots together in subplots (i.e. load prediction/rate plan plots
            together, COP/temp plots together, DC and RN House result plots together)
        subplot_aspect_ratio (float, "maintain", None): The aspect ratio used for parent plots containing subplots,
            "maintain" will preserve aspect_ratio for all subplots
        ax_pad (float): Scaling factor used to determine how far max/min bounds are from actual max/min values on the
            y-axis
        x_ax_tight (bool): Whether the x-axis should be tightened so the axis begins at the first datapoint and ends at
            the last
        save_path (Path): The path to which output for the simulation will be saved
        ext (str): The extension used for plot files
        plot_models (bool): Whether models plots will be generated
        plot_scenario (bool): Whether scenario plots will be generated
        plot_results (bool): Whether results plots will be generated
        soc_combined (bool): Whether SOC plot should contain all controller results as series on a single plot,
            if false all SOC results will be plotted separately
        exclude_controllers_q (str, list[str]): Controller name or list of controller names to exclude from heat flows
            plots
        exclude_controllers_p (str, list[str]): Controller name or list of controller names to exclude from electrical
            power plots
        exclude_controllers_d (str, list[str]): Controller name or list of controller names to exclude from duty cycle
            plots
        show_plots (bool): Whether plots will be displayed using plt.show()
        save_models (bool): Whether models will be saved in file format
        save_scenario (bool): Whether scenario will be saved in file format
        save_results (bool): Whether results will be saved in file format
        save_plots (bool): Whether plots will be saved in file format
        print_to_terminal (bool): Whether output will be printed to the terminal
    """
    def __init__(
            self,
            style: Literal["aps", "nature", "ieee", "prl", "talk"] | None = None,
            heat_color: str | None = None,
            cool_color: str | None = None,
            energy_rate_color: str | None = None,
            demand_rate_color: str | None = None,
            soc_color: str | None = None,
            charge_color: str | None = None,
            disch_color_rn: str | None = None,
            disch_color_dc: str | None = None,
            ashp_color: str | None = None,
            gshp_rn_color: str | None = None,
            gshp_dc_color: str | None = None,
            ashp_aux_color: str | None = None,
            gshp_aux_color: str | None = None,
            tot_color: str | None = None,
            load_color: str | None = None,
            q_color: str | None = None,
            p_color: str | None = None,
            cop_color: str | None = None,
            sec_linestyle: str | None = None,
            flows_style: Literal["line", "stacked_area"] | None = None,
            font_scale: float | None = None,
            linewidth: float | None = None,
            aspect_ratio: float | None = None,
            fig_width: Literal['single', 'double'] | None = None,
            lgd_loc: str | None = None,
            lgd_col: int | Literal["inf"] | None = None,
            lgd_bbox: list | tuple | None = None,
            lgd_frame: bool | None = None,
            lgd_hndl_len: int | None = None,
            grid: bool | None = None,
            pts: int | None = None,
            date_format: str | None = None,
            all_pos: bool | None = None,
            as_subplots: bool | None = None,
            subplot_aspect_ratio: float | Literal["maintain"] | None = None,
            ax_pad: float | None = None,
            x_ax_tight: bool | None = None,
            save_folder: str | Path | None = None,
            save_name: str | None = None,
            ext: str | None = None,
            plot_models: bool | None = None,
            plot_scenario: bool | None = None,
            plot_results: bool | None = None,
            soc_combined: bool | None = None,
            exclude_controllers_q: str | list[str] | None = None,
            exclude_controllers_p: str | list[str] | None = None,
            exclude_controllers_d: str | list[str] | None = None,
            show_plots: bool | None = None,
            save_models: bool | None = None,
            save_scenario: bool | None = None,
            save_results: bool | None = None,
            save_plots: bool | None = None,
            print_to_terminal: bool | None = None,
            scenario: object | None = None):
        """Initializes attributes of the Output class.

        Parameters:
            style: The style used for plotting with paperplot
            heat_color: The color used to represent heating in plots
            cool_color: The color used to represent cooling in plots
            energy_rate_color: The color used to represent energy rates in plots
            demand_rate_color: The color used to represent demand rates in plots
            soc_color: The color used to represent tank state of charge in plots
            charge_color: The color used to represent TES charging in plots
            disch_color_rn: The color used to represent ReNEWW House discharging in plots
            disch_color_dc: The color used to represent DC House discharging in plots
            ashp_color: The color used to represent ASHP in plots
            gshp_rn_color: The color used to represent GSHP to ReNEWW in plots
            gshp_dc_color: The color used to represent GSHP to DC in plots
            ashp_aux_color: The color used to represent ASHP aux heat in plots
            gshp_aux_color: The color used to represent GSHP aux heat in plots
            tot_color: The color used to represent total quantities in plots
            load_color: The color used to represent house thermal load in plots
            q_color: The color used to represent thermal power in plots
            p_color: The color used to represent electrical power in plots
            cop_color: The color used to represent COP in plots
            sec_linestyle: The linestyle used to represent predicted quantities in plots
            flows_style: The style of plot used to represent heat, electrical, and duty cycle
            font_scale: The scaling factor applied to the font size in style
            linewidth: The figure linewidth
            aspect_ratio: The aspect ratio for plots (height/width), defaults to golden ratio
            fig_width: The figure width in columns, 'single', 'double'
            lgd_loc: Setting for ax.legend(loc=_)
            lgd_col: The number of columns in the legend, can be set to 'max' for a horizontal legend
            lgd_bbox: Setting for ax.legend(bbox_to_anchor=_)
            lgd_frame: Whether the legend box is visible
            lgd_hndl_len: Setting for ax.legend(handlelength=_)
            grid: Whether to include the grid on plots
            pts: The number of points used for plotting when applicable
            date_format: Date format used for timeseries plots as strftime string
            all_pos: Whether to take the absolute value of all results for plotting
            as_subplots: Whether to group like plots together in subplots (i.e. load prediction/rate plan plots
                together, COP/temp plots together, DC and RN House result plots together)
            subplot_aspect_ratio: The aspect ratio used for parent plots containing subplots, "maintain" will preserve
                aspect_ratio for all subplots
            ax_pad: Scaling factor used to determine how far max/min bounds are from actual max/min values on the y-axis
            x_ax_tight: Whether the x-axis should be tightened so the axis begins at the first datapoint and ends at the
                last
            save_folder: The parent folder to which output will be saved
            save_name: The name for the child folder to which individual output files will be saved
            ext: The extension used for plot files
            plot_models: Whether models plots will be generated
            plot_scenario: Whether scenario plots will be generated
            plot_results: Whether results plots will be generated
            soc_combined: Whether SOC plot should contain all controller results as series on a single plot,
                if false all SOC results will be plotted separately
            exclude_controllers_q: Controller name or list of controller names to exclude from heat flows plots
            exclude_controllers_p: Controller name or list of controller names to exclude from electrical power plots
            exclude_controllers_d: Controller name or list of controller names to exclude from duty cycle plots
            show_plots: Whether plots will be displayed using plt.show()
            save_models: Whether models will be saved in file format
            save_scenario: Whether scenario will be saved in file format
            save_results: Whether results will be saved in file format
            save_plots: Whether plots will be saved in file format
            print_to_terminal: Whether results will be printed to the terminal
            scenario: An instance of the Scenario class
        """

        # < Set default params >
        if style is None:
            style = "nature"
        # Color cycle for Okabe-Ito is blue (0), vermillion (1), bluish green (2), reddish-purple (3), orange (4),
        # sky blue (5), yellow (6)
        if heat_color is None:
            heat_color = 'C4'
        if cool_color is None:
            cool_color = 'C5'
        if energy_rate_color is None:
            energy_rate_color = 'C0'
        if demand_rate_color is None:
            demand_rate_color = 'C1'
        if soc_color is None:
            soc_color = 'C0'
        if charge_color is None:
            charge_color = 'C0'
        if disch_color_rn is None:
            disch_color_rn = 'C1'
        if disch_color_dc is None:
            disch_color_dc = disch_color_rn
        if ashp_color is None:
            ashp_color = 'C2'
        if gshp_rn_color is None:
            gshp_rn_color = 'C3'
        if gshp_dc_color is None:
            gshp_dc_color = gshp_rn_color
        if ashp_aux_color is None:
            ashp_aux_color = 'C6'
        if gshp_aux_color is None:
            gshp_aux_color = 'C6'
        if tot_color is None:
            tot_color = 'C5'
        if q_color is None:
            q_color = 'C4'
        if p_color is None:
            p_color = 'C0'
        if cop_color is None:
            cop_color = 'C3'
        if load_color is None:
            load_color = 'k'
        if sec_linestyle is None:
            sec_linestyle = '-.'
        if flows_style is None:
            flows_style = 'stacked_area'
        if font_scale is None:
            font_scale = 1
        if fig_width is None:
            fig_width = 'double'
        if lgd_col is None:
            lgd_col = "inf"
        if pts is None:
            pts = 1000
        if date_format is None:
            date_format = "%b %d, %H:%M"
        if all_pos is None:
            all_pos = False
        if as_subplots is None:
            as_subplots = False
        if subplot_aspect_ratio is None:
            subplot_aspect_ratio = aspect_ratio
        if ax_pad is None:
            ax_pad = 1.2
        if x_ax_tight is None:
            x_ax_tight = True
        if save_folder is None:
            save_folder = 'output'
        if scenario is None:
            objective = ""
        else:
            objective = scenario.objective.lower().replace(" ", "_")
        if save_name is None:
            save_name = objective + '_' + dtm.datetime.now().strftime("%Y-%m-%d_%H'%M'%S")
        if ext is None:
            ext = '.pdf'
        if plot_models is None:
            plot_models = True
        if plot_scenario is None:
            plot_scenario = True
        if plot_results is None:
            plot_results = True
        if soc_combined is None:
            soc_combined = False
        if exclude_controllers_q is None:
            exclude_controllers_q = []
        if exclude_controllers_p is None:
            exclude_controllers_p = []
        if exclude_controllers_d is None:
            exclude_controllers_d = []
        if show_plots is None:
            show_plots = True
        if save_models is None:
            save_models = True
        if save_scenario is None:
            save_scenario = True
        if save_results is None:
            save_results = True
        if save_plots is None:
            save_plots = True
        if print_to_terminal is None:
            print_to_terminal = True

        # < Condition params >
        if lgd_bbox is not None:
            lgd_bbox = tuple(lgd_bbox)
        if isinstance(exclude_controllers_q, str):
            exclude_controllers_q = [exclude_controllers_q]
        if isinstance(exclude_controllers_p, str):
            exclude_controllers_p = [exclude_controllers_p]
        if isinstance(exclude_controllers_d, str):
            exclude_controllers_d = [exclude_controllers_d]

        # < Assign class attributes >
        self.style = style
        self.heat_color = heat_color
        self.cool_color = cool_color
        self.energy_rate_color = energy_rate_color
        self.demand_rate_color = demand_rate_color
        self.soc_color = soc_color
        self.charge_color = charge_color
        self.disch_color_rn = disch_color_rn
        self.disch_color_dc = disch_color_dc
        self.ashp_color = ashp_color
        self.gshp_rn_color = gshp_rn_color
        self.gshp_dc_color = gshp_dc_color
        self.ashp_aux_color = ashp_aux_color
        self.gshp_aux_color = gshp_aux_color
        self.tot_color = tot_color
        self.load_color = load_color
        self.q_color = q_color
        self.p_color = p_color
        self.cop_color = cop_color
        self.sec_linestyle = sec_linestyle
        self.flows_style = flows_style
        self.font_scale = font_scale
        self.linewidth = linewidth
        self.aspect_ratio = aspect_ratio
        self.fig_width = fig_width
        self.lgd_loc = lgd_loc
        self.lgd_col = lgd_col
        self.lgd_bbox = lgd_bbox
        self.lgd_frame = lgd_frame
        self.lgd_hndl_len = lgd_hndl_len
        self.grid = grid
        self.pts = pts
        self.date_format = date_format
        self.all_pos = all_pos
        self.as_subplots = as_subplots
        self.subplot_aspect_ratio = subplot_aspect_ratio
        self.ax_pad = ax_pad
        self.x_ax_tight = x_ax_tight
        self.save_path = Path(save_folder) / save_name
        self.ext = ext
        self.plot_models = plot_models
        self.plot_scenario = plot_scenario
        self.plot_results = plot_results
        self.soc_combined = soc_combined
        self.exclude_controllers_q = exclude_controllers_q
        self.exclude_controllers_p = exclude_controllers_p
        self.exclude_controllers_d = exclude_controllers_d
        self.show_plots = show_plots
        self.save_models = save_models
        self.save_scenario = save_scenario
        self.save_results = save_results
        self.save_plots = save_plots
        self.print_to_terminal = print_to_terminal

####################
# Output Functions #
####################


def to_plain(obj, decimals=None, _stack=None):
    """ Recursively convert `obj` -- a scalar, list/dict, numpy array, pandas Series/DataFrame/Index, or (possibly 
    deeply nested) custom class instance -- into a plain structure made only of dict, list, str, int, float, bool, and 
    None. The result is safe to pass to json.dump / yaml.dump. Every float (bare or inside an array/ Series/DataFrame) 
    is rounded to that many decimal places, which is what actually makes the saved file readable rather than full of 
    floating point noise. Raises ValueError if a genuine circular reference is detected."""
    if _stack is None:
        _stack = frozenset()

    # already-plain / JSON-native types (excluding float, handled below)
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj

    # floats (python or numpy) -> rounded, forced to native python float
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return round(value, decimals) if decimals is not None else value

    # other numpy scalars (int8-64, bool_, etc.) -> native python type
    if isinstance(obj, np.generic):
        return obj.item()

    # numpy arrays: round first (if requested), then convert to list
    if isinstance(obj, np.ndarray):
        arr = np.round(obj, decimals) if decimals is not None else obj
        return arr.tolist()

    # pandas Series
    if isinstance(obj, pd.Series):
        return {
            "index": to_plain(obj.index, decimals, _stack),
            "values": to_plain(obj.to_numpy(), decimals, _stack),
        }

    # pandas DataFrame
    if isinstance(obj, pd.DataFrame):
        return {
            "index": to_plain(obj.index, decimals, _stack),
            "columns": obj.columns.tolist(),
            "values": to_plain(obj.to_numpy(), decimals, _stack),
        }

    # pandas datetime-like indices
    if isinstance(obj, pd.DatetimeIndex):
        return [ts.isoformat() for ts in obj]
    if isinstance(obj, pd.TimedeltaIndex):
        return [str(td) for td in obj]
    if isinstance(obj, pd.Index):
        return to_plain(list(obj), decimals, _stack)

    # scalar timestamp / timedelta
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)

    # dict
    if isinstance(obj, dict):
        return {str(k): to_plain(v, decimals, _stack) for k, v in obj.items()}

    # list / tuple / set
    if isinstance(obj, (list, tuple, set)):
        return [to_plain(v, decimals, _stack) for v in obj]

    # custom class instance (possibly nested)
    if (
        hasattr(obj, "__dict__")
        and not inspect.isclass(obj)
        and not inspect.ismodule(obj)
        and not callable(obj)
    ):
        obj_id = id(obj)
        if obj_id in _stack:
            raise ValueError(
                f"Circular reference detected while serializing "
                f"{type(obj).__name__} -- plain JSON/YAML can't represent cycles."
            )
        child_stack = _stack | {obj_id}

        result = {"__class__": type(obj).__name__}
        for attr_name, value in vars(obj).items():
            result[attr_name] = to_plain(value, decimals, child_stack)
        return result

    # last resort: anything else, stringify 
    return str(obj)


def save_json(obj: object, path: str | Path, indent: int =2, decimals: int = 5):
    """Serialize obj to a human-readable JSON file, rounding floats."""
    with open(path, "w") as f:
        json.dump(to_plain(obj, decimals=decimals), f, indent=indent)
        

def save_csv(dct: dict, name: str, output: Output):
    """Saves a dictionary as a csv file in the output folder."""
    d_lists = {k: (list(v) if isinstance(v, (list, tuple, np.ndarray, pd.Series, pd.Index)) else [v])
               for k, v in dct.items()}

    max_len = max(len(v) for v in d_lists.values())

    padded = {
        k: (v + [np.nan] * (max_len - len(v)) if len(v) < max_len else v)
        for k, v in d_lists.items()
    }

    df = pd.DataFrame(padded)
    df.to_csv(output.save_path / (name + ".csv"), index=False)


# def num_series(fig: Figure) -> int:
#     """Calculates the number of series in a figure."""
#     num = 0
#     for ax in fig.axes:
#         num += len(ax.lines)
#         num += len(ax.collections)
#     return num

def num_series(fig: Figure) -> int:
    """Calculates the number of distinct series in a figure and does not double count repeated series."""
    labels = set()
    for ax in fig.axes:
        for artist in list(ax.lines) + list(ax.collections):
            label = artist.get_label()
            if label and not label.startswith('_'):
                labels.add(label)
    return len(labels)


def is_empty(fig: Figure) -> bool:
    """Check if a figure is empty."""
    return num_series(fig) == 0


def style_axes(fig: Figure, o: Output):
    """Styles axes using values from config."""
    if o.linewidth is not None:
        axis_width = o.linewidth*0.7
        tick_width = o.linewidth*0.7
        tick_size = tick_width/0.225
    for ax in fig.axes:
        for spine in ax.spines.values():
            if o.linewidth is not None:
                spine.set_linewidth(axis_width)
                ax.tick_params(width=tick_width, length=tick_size, which="both")
            ax.xaxis.labelpad = 5*o.font_scale
            ax.yaxis.labelpad = 5*o.font_scale


def pp_figure(o: Output) -> tuple[Figure, Axes]:
    """Create a paperplot figure with values from config."""
    pp.use(o.style, font_scale=o.font_scale)
    fig, ax = pp.figure(width=o.fig_width, aspect=o.aspect_ratio)
    if o.linewidth is not None:
        mpl.rcParams['lines.linewidth'] = o.linewidth
    return fig, ax


def pp_subplots(o: Output, size: int | tuple, sharex: bool = False, sharey: bool = False) -> tuple[Figure, list[Axes]]:
    """Create a paperplot subplot with values from config."""
    pp.use(o.style, font_scale=o.font_scale)
    nrows, ncols = size if isinstance(size, tuple) else (size, 1)
    if o.subplot_aspect_ratio is not None:
        if o.subplot_aspect_ratio == "maintain":
            # Maintain aspect ratio of individual plots
            aspect_ratio_sp = o.aspect_ratio*nrows/ncols
        else:
            # use value given
            aspect_ratio_sp = o.subplot_aspect_ratio
    else:
        aspect_ratio_sp = None
    fig, axs = pp.subplots(width=o.fig_width, aspect=aspect_ratio_sp, nrows=nrows, ncols=ncols,
                           sharex=sharex, sharey=sharey)
    if o.linewidth is not None:
        mpl.rcParams['lines.linewidth'] = o.linewidth
    return fig, axs


def fit_legend_ncols(ax, all_lines=None, all_labels=None, max_width_inches=None,
                     min_ncol=1, **legend_kwargs):
    """Draws a legend on `ax`, reducing `ncol` until the legend's rendered width fits within max_width_inches
    (defaults to the figure's own width), or min_ncol is reached. Returns (legend, final_ncol).
    """
    fig = ax.figure
    if max_width_inches is None:
        max_width_inches = fig.get_size_inches()[0]

    if all_lines is None or all_labels is None:
        all_lines, all_labels = ax.get_legend_handles_labels()

    ncol = legend_kwargs.pop('ncol', len(all_labels))
    ncol = max(min_ncol, min(ncol, len(all_labels)))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    legend = None
    while True:
        if legend is not None:
            legend.remove()
        legend = ax.legend(all_lines, all_labels, ncol=ncol, **legend_kwargs)
        fig.canvas.draw()
        width_inches = legend.get_window_extent(renderer=renderer).width / fig.dpi

        if width_inches <= max_width_inches or ncol <= min_ncol:
            break
        ncol -= 1

    return legend, ncol

# def lgd(fig: Figure, ax: Axes, o: Output, all_lines: list[Line2D] | None = None,
#         all_labels: list[str] | None = None):
#     """Creates a legend with the default parameters in the output class instance."""
#     if not is_empty(fig):
#         n_series = num_series(fig)
#         if o.lgd_col == 'inf':
#             n_cols = n_series
#         else:
#             n_cols = min(o.lgd_col, n_series)
#             # Clean up legend rows
#             if (n_cols > 2) and (n_series % (n_cols - 1) == 0):
#                 n_cols = n_cols - 1
#         if (all_lines is not None) and (all_labels is not None):
#             ax.legend(all_lines, all_labels, loc=o.lgd_loc, ncol=n_cols, bbox_to_anchor=o.lgd_bbox, frameon=o.lgd_frame,
#                       handlelength=o.lgd_hndl_len)
#         else:
#             ax.legend(loc=o.lgd_loc, ncol=n_cols, bbox_to_anchor=o.lgd_bbox, frameon=o.lgd_frame,
#                       handlelength=o.lgd_hndl_len)


def lgd(fig: Figure, ax: Axes, o: Output, all_lines: list[Line2D] | None = None,
        all_labels: list[str] | None = None):
    """Creates a legend with the default parameters in the output class instance."""
    if not is_empty(fig):
        n_series = num_series(fig)
        if o.lgd_col == 'inf':
            n_cols = n_series
        else:
            n_cols = min(o.lgd_col, n_series)
            # Reduce column size if legend entries fit cleanly (number of columns between rows is equal)
            if (n_cols > 2) and (n_series % (n_cols - 1) == 0):
                n_cols = n_cols - 1

        legend_kwargs = dict(loc=o.lgd_loc, bbox_to_anchor=o.lgd_bbox, frameon=o.lgd_frame,
                             handlelength=o.lgd_hndl_len)
        # fit_legend_ncols reduces n_cols until legend fits properly on plot
        if (all_lines is not None) and (all_labels is not None):
            fit_legend_ncols(ax, all_lines, all_labels, ncol=n_cols, **legend_kwargs)
        else:
            fit_legend_ncols(ax, ncol=n_cols, **legend_kwargs)


def save_fig(fig: Figure, name: str, output: Output):
    """Saves a figure as the name given and displays in the figure window based on the config."""
    if not is_empty(fig):
        # Set axes style
        style_axes(fig, output)
        if output.show_plots:
            # Show plots in figure window
            fig.canvas.manager.set_window_title(name)
            plt.show()
        if output.save_plots:
            # Save plots in output folder
            fig_name = (name + output.ext)
            print(f"Saving the figure '{fig_name}'...")
            pp.save(fig, output.save_path / fig_name)


def plot_tank_discharge(house: Literal["rn", "dc"], models: object, o: Output,
                        tank_model: Literal["real", "est"] | None = None):
    """Creates plots for tank discharging power."""

    # < Validate params >
    assert tank_model in ["real", "est", None], f"tank_model must be either \"real\" or \"est\"."
    assert house in ["rn", "dc"], f"house must be either \"rn\" or \"dc\"."

    # < Get correct values to plot >
    if tank_model is not None:
        tank = getattr(models, "tank_" + tank_model)
        tank_model = "_" + tank_model
    else:
        # Assume tank models are the same
        tank = models.tank_real
        tank_model = ""

    qd_max = getattr(tank, "qd_max_" + house)
    qd_min = getattr(tank, "qd_min_" + house)
    
    # < Plot >
    # Do not create if discharging is not possible
    if qd_max.any() and qd_min.any():
        fig, ax = pp_figure(o)
        x_space_heat = np.linspace(0, tank.x_max, o.pts)
        x_space_cool = np.linspace(tank.x_min, 0, o.pts)
        ax.plot(x_space_heat, np.maximum(0, np.polyval(qd_max, x_space_heat)), color=o.heat_color, label="Heating")
        ax.plot(x_space_cool, np.minimum(0, np.polyval(qd_min, x_space_cool)), color=o.cool_color, label="Cooling")
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['left'].set_position(('data', 0))
        pp.despine(ax)
        ax.legend(loc='upper left', frameon=o.lgd_frame, handlelength=o.lgd_hndl_len)
        ax.set_xlabel(r"$x$ $(kWh)$")
        ax.set_xlim(np.floor(min([models.tank_real.x_min, models.tank_est.x_min])*o.ax_pad),
                    np.ceil(max([models.tank_real.x_max, models.tank_est.x_max]))*o.ax_pad)
        if house == "rn":
            ax.set_ylabel(r"$q_{d,rn}$ $(kW)$")
        else:
            ax.set_ylabel(r"$q_{d,dc}$ $(kW)$")
        qd_concat = ([np.polyval(models.tank_real.qd_max_rn, models.tank_real.x_max),
                      np.polyval(models.tank_real.qd_max_dc, models.tank_real.x_max),
                      np.polyval(models.tank_est.qd_max_rn, models.tank_real.x_max),
                      np.polyval(models.tank_est.qd_max_dc, models.tank_real.x_max),
                      np.polyval(models.tank_real.qd_min_rn, models.tank_real.x_min),
                      np.polyval(models.tank_real.qd_min_dc, models.tank_real.x_min),
                      np.polyval(models.tank_est.qd_min_rn, models.tank_real.x_min),
                      np.polyval(models.tank_est.qd_min_dc, models.tank_real.x_min)])
        ax.set_ylim(np.floor(min(qd_concat)*o.ax_pad), np.ceil(max(qd_concat)*o.ax_pad))
        plt.grid(o.grid)

        save_fig(fig, "tank" + tank_model + "_discharge_" + house, o)


def plot_load_rate_plan(house: Literal["rn", "dc"], s: object, o: Output):
    """Creates plots for thermal load prediction and rate plan costs."""

    t = s.t
    x_frmt = mdates.DateFormatter(o.date_format)
    house_attrs = getattr(s, house)
    load_real = house_attrs.load_real
    load_pred = forecast_arr_to_vec(house_attrs.load_pred)
    load_concat = np.concatenate((s.rn.load_real, forecast_arr_to_vec(s.rn.load_pred), s.dc.load_real,
                                  forecast_arr_to_vec(s.dc.load_pred)))
    if o.all_pos and (s.rn.load_real <= 0).all() and (forecast_arr_to_vec(s.rn.load_pred) <= 0).all() and (
            s.dc.load_real <= 0).all() and (forecast_arr_to_vec(s.dc.load_pred) <= 0).all():
        load_real = np.abs(load_real)
        load_pred = np.abs(load_pred)
        load_concat = np.abs(load_concat)
    energy_rate_baseline = house_attrs.rate_plan_baseline.energy_rate
    energy_rate_other = house_attrs.rate_plan_other.energy_rate
    demand_rate_baseline = house_attrs.rate_plan_baseline.demand_rate
    demand_rate_other = house_attrs.rate_plan_other.demand_rate

    fig, ax = pp_figure(o)

    ax.plot(t, load_real, color=o.load_color, label='Real Load')
    ax.plot(t, load_pred, color=o.load_color, linestyle=o.sec_linestyle, label='Predicted Load')
    plt.xticks(rotation=30, ha='right')
    ax.autoscale(enable=True, axis='x', tight=o.x_ax_tight)
    ax.xaxis.set_major_formatter(x_frmt)
    ax.set_ylabel('Thermal Load (kW)')
    ax.set_ylim(min(0, np.floor(np.min(load_concat)*o.ax_pad)), max(0, np.ceil(np.max(load_concat)*o.ax_pad)))
    plt.grid(o.grid)

    ax2 = ax.twinx()

    if (energy_rate_baseline == energy_rate_other).all():
        ax2.step(t, energy_rate_baseline, color=o.energy_rate_color,
                 label='Energy Rate: ' + house_attrs.rate_plan_baseline.name, where='post')
    else:
        ax2.step(t, energy_rate_baseline, color=o.energy_rate_color,
                 linestyle=o.sec_linestyle, label='Baseline Energy Rate: ' + house_attrs.rate_plan_baseline.name,
                 where='post')
        ax2.step(t, energy_rate_other, color=o.energy_rate_color,
                 label='Main Energy Rate: ' + house_attrs.rate_plan_other.name, where='post')
    ax2.autoscale(enable=True, axis='x', tight=o.x_ax_tight)
    ax2.set_ylabel('Energy Price ($/kWh)')
    ax2.set_ylim(0, np.round(max(np.mean(energy_rate_baseline), np.mean(energy_rate_other)*2), 1))
    ax2.tick_params(axis='y', colors=o.energy_rate_color)
    ax2.yaxis.label.set_color(o.energy_rate_color)
    ax2.spines['right'].set_color(o.energy_rate_color)

    if demand_rate_baseline.any() or demand_rate_other.any():
        ax3 = ax.twinx()

    if demand_rate_baseline.any():
        if (demand_rate_baseline == demand_rate_other).all():
            ax3.step(t, demand_rate_baseline, color=o.demand_rate_color,
                     label='Demand Rate: ' + house_attrs.rate_plan_baseline.name, where='post')
        else:
            ax3.step(t, demand_rate_baseline, color=o.demand_rate_color,
                     linestyle=o.sec_linestyle, label='Baseline Demand Rate: ' + house_attrs.rate_plan_baseline.name,
                     where='post')
    if demand_rate_other.any():
        if (demand_rate_baseline != demand_rate_other).any():
            ax3.step(t, demand_rate_other, color=o.demand_rate_color,
                     label='Main Demand Rate: ' + house_attrs.rate_plan_other.name, where='post')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if demand_rate_baseline.any() or demand_rate_other.any():
        ax3.autoscale(enable=True, axis='x', tight=o.x_ax_tight)
        ax3.set_ylabel('Demand Price ($/kW)')
        ax3.set_ylim(0, np.round(max(np.mean(demand_rate_baseline), np.mean(demand_rate_other)*2), 1))
        ax3.tick_params(axis='y', colors=o.demand_rate_color)
        ax3.yaxis.label.set_color(o.demand_rate_color)
        ax3.spines['right'].set_color(o.demand_rate_color)
        ax3.spines['right'].set_position(('outward', 60))
        lines3, labels3 = ax3.get_legend_handles_labels()
        all_lines = lines1 + lines2 + lines3
        all_labels = labels1 + labels2 + labels3
    else:
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2
    # Change lgd_col so plot is presented properly
    o_copy = deepcopy(o)
    if isinstance(o_copy.lgd_col, float | int):
        o_copy.lgd_col = min(o_copy.lgd_col, 2)
    lgd(fig, ax, o, all_lines, all_labels)

    save_fig(fig, "thermal_load_rate_plan_" + house, o)


def plot_temp_cop(s: object, o: Output):
    """Creates plots for source temperature and COP."""

    # < Setup >
    t = s.t
    n = len(t)
    # Heating load indices for ReNEWW House
    h_rn_real = np.where(s.rn.load_real >= 0)[0]
    h_rn_pred = np.where(s.rn.load_pred >= 0)[0]
    # Cooling load indices for ReNEWW House
    c_rn_real = np.where(s.rn.load_real < 0)[0]
    c_rn_pred = np.where(s.rn.load_pred < 0)[0]
    # Heating load indices for DC House
    h_dc_real = np.where(s.dc.load_real >= 0)[0]
    h_dc_pred = np.where(s.dc.load_pred >= 0)[0]
    # Cooling load indices for DC House
    c_dc_real = np.where(s.dc.load_real < 0)[0]
    c_dc_pred = np.where(s.dc.load_pred < 0)[0]
    # Outdoor air temperature arrays
    theta_pred = forecast_arr_to_vec(s.theta_pred)
    # COP arrays
    cop_rn_real = np.full(n, np.nan)
    cop_rn_real[h_rn_real] = s.rn.cop_heat_real[h_rn_real]
    cop_rn_real[c_rn_real] = s.rn.cop_cool_real[c_rn_real]
    cop_rn_real = np.abs(cop_rn_real)
    cop_rn_pred = np.full(n, np.nan)
    cop_rn_pred[h_rn_pred] = forecast_arr_to_vec(s.rn.cop_heat_pred)[h_rn_pred]
    cop_rn_pred[c_rn_pred] = s.rn.cop_cool_pred[0, c_rn_pred]
    cop_rn_pred = np.abs(cop_rn_pred)
    cop_dc_real = np.full(n, np.nan)
    cop_dc_real[h_dc_real] = s.dc.cop_heat_real[h_dc_real]
    cop_dc_real[c_dc_real] = s.dc.cop_cool_real[c_dc_real]
    cop_dc_real = np.abs(cop_dc_real)
    cop_dc_pred = np.full(n, np.nan)
    cop_dc_pred[h_dc_pred] = forecast_arr_to_vec(s.dc.cop_heat_pred)[h_dc_pred]
    cop_dc_pred[c_dc_pred] = forecast_arr_to_vec(s.dc.cop_cool_pred)[c_dc_pred]
    cop_dc_pred = np.abs(cop_dc_pred)

    if o.all_pos:
        cop_rn_real = np.abs(cop_rn_real)
        cop_rn_pred = np.abs(cop_rn_pred)
        cop_dc_real = np.abs(cop_dc_real)
        cop_dc_pred = np.abs(cop_dc_pred)

    x_frmt = mdates.DateFormatter(o.date_format)

    # < ReNEWW House Plot >
    fig, ax = pp_figure(o)

    ax.plot(t, s.temp_egl, color='k', label='Real Temperature')
    plt.xticks(rotation=30, ha='right')
    ax.xaxis.set_major_formatter(x_frmt)
    ax.autoscale(enable=True, axis='x', tight=o.x_ax_tight)
    ax.set_ylabel('Entering Ground Loop\n' + r'Temperature ($\degree$ C)')
    if h_rn_real.any() and not c_rn_real.any():
        egl_bnds = [6, 16]
    elif c_rn_real.any() and not h_rn_real.any():
        egl_bnds = [10, 25]
    else:
        egl_bnds = [8, 18]
    ax.set_ylim((min(egl_bnds[0], np.floor(np.min(s.temp_egl)/o.ax_pad)), max(egl_bnds[1],
                                                                              np.ceil(np.max(s.temp_egl)*o.ax_pad))))
    plt.grid(o.grid)

    ax2 = ax.twinx()

    ax2.plot(t, cop_rn_real, color=o.gshp_rn_color, label='Real COP')
    ax2.plot(t, cop_rn_pred, color=o.gshp_rn_color, linestyle=o.sec_linestyle, label='Predicted COP')
    ax2.xaxis.set_major_formatter(x_frmt)
    ax2.autoscale(enable=True, axis='x', tight=o.x_ax_tight)
    ax2.set_ylabel(r'Coefficient of Performance')
    cop_concat = np.concatenate((cop_rn_real, cop_rn_pred, cop_dc_real, cop_dc_pred))
    y2lim = (min(1, np.floor(np.min(cop_concat))), max(1, np.ceil(np.max(cop_concat))))
    ax2.set_ylim(y2lim)
    ax2.tick_params(axis='y', colors=o.gshp_rn_color)
    ax2.yaxis.label.set_color(o.gshp_rn_color)
    ax2.spines['right'].set_color(o.gshp_rn_color)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    all_lines = lines1 + lines2
    all_labels = labels1 + labels2
    lgd(fig, ax, o, all_lines, all_labels)

    save_fig(fig, "cop_gshp_temp_egl", o)

    # < DC House Plot >
    fig, ax = pp_figure(o)

    ax.plot(t, s.theta_real, color='k', label='Real Temperature')
    ax.plot(t, theta_pred, color='k', linestyle=o.sec_linestyle, label='Predicted Temperature')
    plt.xticks(rotation=30, ha='right')
    ax.xaxis.set_major_formatter(x_frmt)
    ax.autoscale(enable=True, axis='x', tight=o.x_ax_tight)
    ax.set_ylabel(r'Outdoor Air Temperature ($\degree$ C)')
    if h_rn_real.any() and not c_rn_real.any():
        theta_bnds = [-20, 0]
    elif c_rn_real.any() and not h_rn_real.any():
        theta_bnds = [20, 35]
    else:
        theta_bnds = [8, 20]
    theta_concat = np.concatenate((s.theta_real, theta_pred))
    ax.set_ylim((min(theta_bnds[0], np.floor(np.min(theta_concat))), max(theta_bnds[1], np.ceil(np.max(theta_concat)))))
    plt.grid(o.grid)

    ax2 = ax.twinx()

    ax2.plot(t, cop_dc_real, color=o.ashp_color, label='Real COP')
    ax2.plot(t, cop_dc_pred, color=o.ashp_color, linestyle=o.sec_linestyle, label='Predicted COP')
    ax2.xaxis.set_major_formatter(x_frmt)
    ax2.autoscale(enable=True, axis='x', tight=o.x_ax_tight)
    ax2.set_ylabel(r'Coefficient of Performance')
    ax2.set_ylim(y2lim)
    ax2.tick_params(axis='y', colors=o.ashp_color)
    ax2.yaxis.label.set_color(o.ashp_color)
    ax2.spines['right'].set_color(o.ashp_color)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    all_lines = lines1 + lines2
    all_labels = labels1 + labels2
    lgd(fig, ax, o, all_lines, all_labels)

    save_fig(fig, "cop_ashp_theta", o)


def plot_soc(controllers: str | list[str], m: object, r: dict, o: Output):
    """Plots the tank state of charge for the simulation, o.soc_combined determines whether all controllers are plotted
    in a single figure or on seperate figures."""

    if o.soc_combined:
        t = []
        x = []
        cs = []
        x_concat = np.ndarray([])
        for controller in controllers:
            r_c = r[controller]
            if r_c['x'].any():
                cs.append(controller)
                t.append(r_c['t'])
                x.append(r_c['x'])
                x_concat = np.concatenate(tuple(x))
    else:
        t = r['t']
        x = r['x']
        x_concat = x
    t_frmt = mdates.DateFormatter(o.date_format)

    if o.all_pos and (x_concat <= 0).all():
        if isinstance(x, list):
            for i, x_i in enumerate(x):
                x[i] = np.abs(x_i)
        else:
            x = np.abs(x)
        ylim = [0, np.abs(m.tank_real.x_min)]
    else:
        if (x_concat > 0).any():
            ylim_top = m.tank_real.x_max
        else:
            ylim_top = 0
        if (x_concat <= 0).any():
            ylim_bottom = m.tank_real.x_min
        else:
            ylim_bottom = 0
        ylim = [ylim_bottom, ylim_top]

    fig, ax = pp_figure(o)

    if o.soc_combined:
        for i, c in enumerate(cs):
            if c == "Omniscient Optimal":
                ax.plot(t[i], x[i], label=c, color='C2')
            elif c == "Causal MPC":
                ax.plot(t[i], x[i], label=c, color='C1')
            else:
                ax.plot(t[i], x[i], label=c)
        lgd(fig, ax, o)
        fig_name = "soc"
    else:
        ax.plot(t, x, color=o.soc_color)
        fig_name = controllers.lower().replace(" ", "_") + "_soc"
    plt.xticks(rotation=30, ha='right')
    ax.xaxis.set_major_formatter(t_frmt)
    ax.autoscale(enable=True, axis='x', tight=o.x_ax_tight)
    ax.set_ylabel('Stored Tank Energy (kWh)')
    ax.set_ylim(ylim)
    plt.grid(o.grid)

    save_fig(fig, fig_name, o)


def plot_flows_line(controller: str, r: dict, val: Literal["q", "p", "d"], o: Output, sp_rn: Figure | None = None,
                    sp_dc: Figure | None = None, sp_num: int | None = None, sp_tot: int | None = None):
    """Plots thermal heat flows, electrical power, or duty cycle as line plots for controller simulation."""

    # < Setup >
    if val == "q":
        ylabel = 'Thermal Power (kW)'
    elif val == "p":
        ylabel = 'Electrical Power (kW)'
    else:
        ylabel = 'Duty Cycle'
    t = r['t']
    x_frmt = mdates.DateFormatter(o.date_format)

    # < ReNEWW House flows >
    fig, ax = pp_figure(o)

    if r[val + 'd_rn'].any():
        ax.plot(t, r[val + 'd_rn'], color=o.disch_color_rn, label='TES Discharging')
    if r[val + 'c'].any():
        ax.plot(t, r[val + 'c'], color=o.charge_color, label='GSHP TES Charging')
    if r[val + '_gshp_rn'].any():
        ax.plot(t, r[val + '_gshp_rn'], color=o.gshp_rn_color, label='GSHP to Space')
    if val != "q":
        if r[val + '_gshp_dc'].any():
            ax.plot(t, r[val + '_gshp_dc'], color=o.gshp_dc_color, label='GSHP to DC')
    if val == "d":
        if r['d_aux_gshp'].any():
            ax.plot(t, r['d_aux_gshp'], color=o.gshp_aux_color, label='Aux Heat')
    else:
        if r['p_aux_gshp'].any():
            ax.plot(t, r['p_aux_gshp'], color=o.gshp_aux_color, label='Aux Heat')
    if (val == "q") and r['load_rn'].any():
        if not np.allclose(r['q_tot_del_rn'], r['q_gshp_rn']):
            ax.plot(t, r['q_tot_del_rn'], color=o.tot_color, label='Total to Space')
        # Use secondary linestyle to differentiate
        ax.plot(t, r['load_rn'], color=o.load_color, linestyle=o.sec_linestyle, label='Load')
    lgd(fig, ax, o)
    plt.xticks(rotation=30, ha='right')
    ax.xaxis.set_major_formatter(x_frmt)
    ax.set_ylabel(ylabel)
    if val =="d":
        ylim = [0, 1]
    else:
        val_concat = np.concatenate((r[val + 'd_rn'], r[val + 'c'], r[val + '_gshp_rn'], r['p_aux_gshp'],
                                     r[val + 'd_dc'], r[val + '_ashp'], r['p_aux_ashp']))
        if (val == "q"):
            val_concat = np.concatenate((val_concat, r['load_rn'], r['load_dc']))
        ylim = (min(0, np.floor(np.min(val_concat)*o.ax_pad)), max(0, np.ceil(np.max(val_concat)*o.ax_pad)))
    ax.set_ylim(ylim)
    plt.grid(o.grid)

    if val == "q":
        save_fig(fig, controller.lower().replace(" ", "_") + "_heat_flows_rn", o)
    elif val == "p":
        save_fig(fig, controller.lower().replace(" ", "_") + "_elec_power_rn", o)
    else:
        save_fig(fig, controller.lower().replace(" ", "_") + "_duty_cycle_rn", o)

    # < DC House flows >
    fig, ax = pp_figure(o)

    if r[val + 'd_dc'].any():
        ax.plot(t, r[val + 'd_dc'], color=o.disch_color_dc, label='TES Discharging')
    if r[val + '_ashp'].any():
        ax.plot(t, r[val + '_ashp'], color=o.ashp_color, label='ASHP')
    if val == "d":
        if r['d_aux_ashp'].any():
            ax.plot(t, r['d_aux_ashp'], color=o.ashp_aux_color, label='Aux Heat')
    else:
        if r['p_aux_ashp'].any():
            ax.plot(t, r['p_aux_ashp'], color=o.ashp_aux_color, label='Aux Heat')
    if val == "q":
        if r[val + '_gshp_dc'].any():
            ax.plot(t, r[val + '_gshp_dc'], color=o.gshp_dc_color, label='GSHP to Space')
    if (val == "q") and r['load_dc'].any():
        if not np.allclose(r['q_tot_del_dc'], r['q_ashp']):
            ax.plot(t, r['q_tot_del_dc'], color=o.tot_color, label='Total to Space')
        # Use secondary linestyle to differentiate
        ax.plot(t, r['load_dc'], color=o.load_color, linestyle=o.sec_linestyle, label='Load')
    lgd(fig, ax, o)
    plt.xticks(rotation=30, ha='right')
    ax.xaxis.set_major_formatter(x_frmt)
    ax.autoscale(enable=True, axis='x', tight=o.x_ax_tight)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    plt.grid(o.grid)

    if val == "q":
        save_fig(fig, controller.lower().replace(" ", "_") + "_heat_flows_dc", o)
    elif val == "p":
        save_fig(fig, controller.lower().replace(" ", "_") + "_elec_power_dc", o)
    else:
        save_fig(fig, controller.lower().replace(" ", "_") + "_duty_cycle_dc", o)


def plot_flows_stacked_area(controller: str, r: dict, val: Literal["q", "p", "d"], o: Output,
                            sp_rn: tuple[Figure, list[Axes]] | None = None,
                            sp_dc: tuple[Figure, list[Axes]] | None = None, sp_num: int | None = None,
                            sp_tot: int | None = None):
    """Plots thermal heat flows, electrical power, or duty cycle as stacked area plots for controller simulation."""

    # < Setup >
    if val == "q":
        ylabel = 'Thermal Power (kW)'
    elif val == "p":
        ylabel = 'Electrical Power (kW)'
    else:
        ylabel = 'Duty Cycle'
    t = r['t']
    x_frmt = mdates.DateFormatter(o.date_format)

    # Series descriptors as dicts
    series_d_rn = {'data': r[val + 'd_rn'], 'color': o.disch_color_rn, 'label': 'TES Discharging to RN'}
    series_d_dc = {'data': r[val + 'd_dc'], 'color': o.disch_color_dc, 'label': 'TES Discharging to DC'}
    series_c = {'data': r[val + 'c'], 'color': o.charge_color, 'label': 'GSHP TES Charging'}
    series_gshp_rn = {'data': r[val + '_gshp_rn'], 'color': o.gshp_rn_color, 'label': 'GSHP to RN'}
    series_gshp_dc = {'data': r[val + '_gshp_dc'], 'color': o.gshp_dc_color, 'label': 'GSHP to DC'}
    if val == "q":
        val_adj = "p"
    else:
        val_adj = val
    series_gshp_aux = {'data': r[val_adj + '_aux_gshp'], 'color': o.gshp_aux_color, 'label': 'GSHP Aux Heat'}
    series_ashp = {'data': r[val + '_ashp'], 'color': o.ashp_color, 'label': 'ASHP'}
    series_ashp_aux = {'data': r[val_adj + '_aux_ashp'], 'color': o.ashp_aux_color, 'label': 'ASHP Aux Heat'}
    series_load_rn = {'data': r['load_rn'], 'color': o.load_color, 'label': 'Load'}
    series_load_dc = {'data': r['load_dc'], 'color': o.load_color, 'label': 'Load'}

    if val == "q":
        rn_series = [series_d_rn, series_gshp_rn, series_gshp_aux, series_load_rn, series_c]
        dc_series = [series_d_dc, series_gshp_dc, series_ashp, series_ashp_aux, series_load_dc]
        rn_as_line = [series_load_rn, series_c]
        dc_as_line = [series_load_dc]
    elif val == 'p':
        rn_series = [series_d_rn, series_gshp_rn, series_gshp_dc, series_gshp_aux, series_c]
        dc_series = [series_d_dc, series_ashp, series_ashp_aux]
        rn_as_line = []
        dc_as_line = []
    else:
        rn_series = [series_gshp_rn, series_gshp_dc, series_gshp_aux, series_c]
        dc_series = [series_ashp, series_ashp_aux]
        rn_as_line = [series_gshp_aux]
        dc_as_line = [series_ashp, series_ashp_aux]

    # < ReNEWW House series collection >
    data_rn = []
    colors_rn = []
    labels_rn = []
    for dct in rn_series:
        if dct['data'].any() and not any(dct is x for x in rn_as_line):
            data_rn.append(dct['data'])
            colors_rn.append(dct['color'])
            labels_rn.append(dct['label'])

    # < DCHouse House series collection >
    data_dc = []
    colors_dc = []
    labels_dc = []
    for dct in dc_series:
        if dct['data'].any() and not any(dct is x for x in dc_as_line):
            data_dc.append(dct['data'])
            colors_dc.append(dct['color'])
            labels_dc.append(dct['label'])

    # < ReNEWW House flows >
    if o.as_subplots:
        assert (sp_rn is not None) and (sp_dc is not None) and (sp_num is not None) and (sp_tot is not None), \
            "sp_rn, sp_dc, sp_num, and sp_tot must be included to plot flows as subplots."
        fig_rn = sp_rn[0]
        ax = sp_rn[1][sp_num]
        ax.set_title(chr(ord('`')+sp_num+1) + ") " + controller)
    else:
        fig_rn, ax = pp_figure(o)

    if len(data_rn) != 0:
        ax.stackplot(t, tuple(data_rn), colors=colors_rn, labels=labels_rn)
    for dct in rn_as_line:
        if dct['data'].any():
            ax.plot(t, dct['data'], color=dct['color'], label=dct['label'])
    if (not o.as_subplots) or (o.as_subplots and sp_num + 1 == sp_tot):
        # Only plot legend for individual plots or last subplot
        lgd(fig_rn, ax, o)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha('right')
    ax.xaxis.set_major_formatter(x_frmt)
    ax.autoscale(enable=True, axis='x', tight=o.x_ax_tight)
    ax.set_ylabel(ylabel)
    if val == "d":
        ylim = [0, 1]
    else:
        if val == "q":
            val_concat = np.concatenate((r['qc'], r['load_rn'], r['load_dc']))
        else:
            val_concat = np.concatenate((sum(data_rn), sum(data_dc)))
        ylim = (min(0, np.floor(np.min(val_concat)*o.ax_pad)), max(0, np.ceil(np.max(val_concat)*o.ax_pad)))
    ax.set_ylim(ylim)
    plt.grid(o.grid)

    if not o.as_subplots:
        # Save figure separately
        if val == "q":
            save_fig(fig_rn, controller.lower().replace(" ", "_") + "_heat_flows_rn", o)
        elif val == "p":
            save_fig(fig_rn, controller.lower().replace(" ", "_") + "_elec_power_rn", o)
        else:
            save_fig(fig_rn, controller.lower().replace(" ", "_") + "_duty_cycle_rn", o)
    elif sp_num + 1 == sp_tot:
        # Ensure only one axis has labels if shared axis enabled
        pp.clean_shared_axes(fig_rn)
        # Save figure when final subplot has been plotted
        if val == "q":
            save_fig(fig_rn, "heat_flows_rn", o)
        elif val == "p":
            save_fig(fig_rn, "elec_power_rn", o)
        else:
            save_fig(fig_rn, "duty_cycle_rn", o)

    # < DC House flows >
    if o.as_subplots:
        fig_dc = sp_dc[0]
        ax = sp_dc[1][sp_num]
        ax.set_title(chr(ord('`')+sp_num+1) + ") " + controller)
    else:
        fig_dc, ax = pp_figure(o)

    if len(data_dc) != 0:
        ax.stackplot(t, tuple(data_dc), colors=colors_dc, labels=labels_dc)
    for dct in dc_as_line:
        if dct['data'].any():
            ax.plot(t, dct['data'], color=dct['color'], label=dct['label'])
    if (not o.as_subplots) or (o.as_subplots and sp_num + 1 == sp_tot):
        # Only plot legend for individual plots or last subplot
        lgd(fig_dc, ax, o)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha('right')
    ax.xaxis.set_major_formatter(x_frmt)
    ax.autoscale(enable=True, axis='x', tight=o.x_ax_tight)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    plt.grid(o.grid)

    if not o.as_subplots:
        # Ensure only one axis has labels if shared axis enabled
        pp.clean_shared_axes(fig_dc)
        # Save figure separately
        if val == "q":
            save_fig(fig_dc, controller.lower().replace(" ", "_") + "_heat_flows_dc", o)
        elif val == "p":
            save_fig(fig_dc, controller.lower().replace(" ", "_") + "_elec_power_dc", o)
        else:
            save_fig(fig_dc, controller.lower().replace(" ", "_") + "_duty_cycle_dc", o)
    elif sp_num + 1 == sp_tot:
        # Save figure when final subplot has been plotted
        if val == "q":
            save_fig(fig_dc, "heat_flows_dc", o)
        elif val == "p":
            save_fig(fig_dc, "elec_power_dc", o)
        else:
            save_fig(fig_dc, "duty_cycle_dc", o)


def plot_tuning(indep: list | np.ndarray, indep_name: str, elec_cost_savings: list | np.ndarray,  o: Output,
                peak_demand_red: list | np.ndarray | None = None, solve_time: list | np.ndarray | None = None):
    """Create tuning plots."""

    fig, ax = pp_figure(o)

    ax.plot(indep, elec_cost_savings, color=o.energy_rate_color, label='Electricity Cost')
    if peak_demand_red is not None:
        ax.plot(indep, peak_demand_red, color=o.demand_rate_color, label='Peak Demand')
    ax.set_xlabel(indep_name)
    ax.autoscale(enable=True, axis='x', tight=o.x_ax_tight)
    if peak_demand_red is None:
        ax.set_ylabel('Electricity Cost Reduction (%)')

    else:
        ax.set_ylabel('Objective Improvement (%)')
        lgd(fig, ax, o)
    ax.set_xticks(np.arange(0, indep[-1], 2))
    ax.set_ylim(bottom=0)
    plt.grid(o.grid)

    if solve_time is not None:
        ax2 = ax.twinx()
        ax2.plot(indep, solve_time, color='C3')
        ax2.autoscale(enable=True, axis='x', tight=o.x_ax_tight)
        ax2.set_ylabel('Computation Time (s)')
        ax2.set_ylim(bottom=0)
        ax2.tick_params(axis='y', colors='C3')
        ax2.yaxis.label.set_color('C3')
        ax2.spines['right'].set_color('C3')

    save_fig(fig, indep_name.lower().replace(" ", "_").replace("(", "").replace(")", "") + "_tuning", o)


def generate_models_output(models: object, output: Output, config_files: dict):
    """Generates the models output."""
    if output.save_models:
        # Save scenario config as .toml
        print(f"Saving models config as '{config_files['models']}'...")
        copy_config(config_files['models'], output.save_path)
        # Save models attributes to json
        print(f"Saving models class as 'models.json'...")
        save_json(models, output.save_path / "models.json")
    if output.plot_models:
        # Generate tank discharging power plots
        if np.array_equal(models.tank_real.qd_max_rn, models.tank_est.qd_max_rn
        ) and np.array_equal(models.tank_real.qd_min_rn, models.tank_est.qd_min_rn
        ) and np.array_equal(models.tank_real.qd_max_dc, models.tank_est.qd_max_dc
        ) and np.array_equal(models.tank_real.qd_min_dc, models.tank_est.qd_min_dc
        ) and np.array_equal(models.tank_real.x_max, models.tank_est.x_max
        ) and np.array_equal(models.tank_real.x_min, models.tank_est.x_min
        ):
            # Tank models are the same
            plot_tank_discharge("rn", models, output)
            plot_tank_discharge("dc", models, output)
        else:
            # Tank models are different
            plot_tank_discharge("rn", models, output, "real")
            plot_tank_discharge("dc", models, output, "real")
            plot_tank_discharge("rn", models, output, "est")
            plot_tank_discharge("dc", models, output, "est")


def generate_scenario_output(scenario: object, output: Output, config_files: dict):
    """Generates the scenario output."""
    if output.save_scenario:
        # Save scenario config as .toml
        print(f"Saving scenario config as '{config_files['output']}'...")
        copy_config(config_files['output'], output.save_path)
        # Save scenario attributes to json
        print(f"Saving scenario class as 'scenario.json'...")
        save_json(scenario, output.save_path / "scenario.json")
    if output.plot_scenario:
        plot_load_rate_plan("rn", scenario, output)
        plot_load_rate_plan("dc", scenario, output)


def generate_results_output(controller: str, models: object, results: dict, output: Output):
    """Generates the results output."""
    if output.plot_results and results['x'].any() and (not output.soc_combined):
        plot_soc(controller, models, results, output)

    if output.save_results:
        # Save results to .csv
        controller_name = controller.lower()
        save_name = controller_name.replace(" ", "_") + "_results"
        print(f"Saving results for {controller_name} controller as {save_name}.csv...")
        save_csv(results, save_name, output)


def print_output(scenario: object, results: dict, output: Output):
    """Prints output metrics to terminal."""
    if output.print_to_terminal:
        print(f"\n============== {scenario.objective} Metrics ==============")

    for controller in scenario.controllers:
        r = results[controller]
        print(f"\n------ {controller} ------")
        print(f"Objective electricity cost: ${r['elec_cost_obj']}")
        if not np.isnan(r['perc_cost_savings_baseline_no_tes']):
            print(f"Cost savings with reference to 'No TES Baseline': ${r['cost_savings_baseline_no_tes']}, "
                  f"{r['perc_cost_savings_baseline_no_tes']}%")
        if not np.isnan(r['perc_cost_savings_baseline_tes']):
            print(f"Cost savings with reference to 'RBC TES Baseline': ${r['cost_savings_baseline_tes']}, "
                  f"{r['perc_cost_savings_baseline_tes']}%")
        if not np.isnan(r['perc_peak_red_rn_baseline_no_tes']):
            print(f"ReNEWW House on-peak demand reduction with reference to 'No TES Baseline': "
                  f"{r['perc_peak_red_rn_baseline_no_tes']}%")
        if not np.isnan(r['perc_peak_red_dc_baseline_no_tes']):
            print(f"DC House on-peak demand reduction with reference to 'No TES Baseline': "
                  f"{r['perc_peak_red_dc_baseline_no_tes']}%")
        if not np.isnan(r['perc_peak_red_tot_baseline_no_tes']):
            print(f"Total on-peak demand reduction with reference to 'No TES Baseline': "
                  f"{r['perc_peak_red_tot_baseline_no_tes']}%")
        if not np.isnan(r['perc_peak_red_rn_baseline_tes']):
            print(f"ReNEWW House on-peak demand reduction with reference to 'RBC TES Baseline': "
                  f"{r['perc_peak_red_rn_baseline_tes']}%")
        if not np.isnan(r['perc_peak_red_dc_baseline_tes']):
            print(f"DC House on-peak demand reduction with reference to 'RBC TES Baseline': "
                  f"{r['perc_peak_red_dc_baseline_tes']}%")
        if not np.isnan(r['perc_peak_red_tot_baseline_tes']):
            print(f"Total on-peak demand reduction with reference to 'RBC TES Baseline': "
                  f"{r['perc_peak_red_tot_baseline_tes']}%")
        if (r['mean_discomf_rn'] == 0) and (r['mean_discomf_dc'] == 0):
            print("No discomfort")
        else:
            print(f"Average discomfort at ReNEWW House: {r['mean_discomf_rn']} kW")
            print(f"Average discomfort at DC House: {r['mean_discomf_dc']} kW")


def generate_output(models: object, scenario: object, results: dict, config_files: dict):
    """Generates all outputs based on the settings in the output config."""

    output = instantiate_class_from_toml(Output, config_files['output'], add_attr={'scenario': scenario})
    if output.save_models or output.save_scenario or output.save_results or output.save_plots:
        print(f"Creating the directory '{output.save_path}' to save output...")
        output.save_path.mkdir(parents=True, exist_ok=True)

    generate_models_output(models, output, config_files)
    generate_scenario_output(scenario, output, config_files)
    plot_temp_cop(scenario, output)
    controllers = scenario.controllers
    if output.soc_combined:
        # Plot all controllers SOC on a single plot
        plot_soc(controllers, models, results, output)
    if output.as_subplots:
        # < Create figures for flow subplots >
        # Total number of controllers
        n_c = len(controllers)
        # Number of non-excluded controllers in for each subplot
        n_c_q = n_c - len(set(output.exclude_controllers_q) & set(controllers))
        n_c_p = n_c - len(set(output.exclude_controllers_p) & set(controllers))
        n_c_d = n_c - len(set(output.exclude_controllers_d) & set(controllers))
        # Create subplots with correct number of controllers
        # sp contains tuple of (fig, axs)
        sp_rn_q = pp_subplots(output, n_c_q, True)
        sp_rn_p = pp_subplots(output, n_c_p, True)
        sp_rn_d = pp_subplots(output, n_c_d, True)
        sp_dc_q = pp_subplots(output, n_c_q, True)
        sp_dc_p = pp_subplots(output, n_c_p, True)
        sp_dc_d = pp_subplots(output, n_c_d, True)
        # Initiate counters for number of exclusions
        num_excl_q = 0
        num_excl_p = 0
        num_excl_d = 0
    for i, controller in enumerate(controllers):
        if output.all_pos and (scenario.rn.load_real <= 0).all() and (scenario.dc.load_real <= 0).all():
            controller_results_plot = pos_dict(results[controller])
        else:
            controller_results_plot = results[controller]
        generate_results_output(controller, models, results[controller], output)
        # Choose correct plotting function for flows
        plot_flows = globals()["plot_flows_" + output.flows_style]
        # Choose function arguments based on whether subplots are to be used
        if output.as_subplots:
            if controller not in output.exclude_controllers_q:
                plot_flows(controller, controller_results_plot, "q", output, sp_rn_q, sp_dc_q, i - num_excl_q, n_c_q)
            else:
                num_excl_q += 1
            if controller not in output.exclude_controllers_p:
                plot_flows(controller, controller_results_plot, "p", output, sp_rn_p, sp_dc_p, i - num_excl_p, n_c_p)
            else:
                num_excl_p += 1
            if controller not in output.exclude_controllers_d:
                plot_flows(controller, controller_results_plot, "d", output, sp_rn_d, sp_dc_d, i - num_excl_d, n_c_d)
            else:
                num_excl_d += 1
        else:
            plot_flows(controller, controller_results_plot, "q", output)
            plot_flows(controller, controller_results_plot, 'p', output)
            plot_flows(controller, controller_results_plot, "d", output)

    print_output(scenario, results, output)
