import argparse
import data
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import model
import numpy as np
from numpy.polynomial import polynomial
import os
import pandas
import plots
import utils

def get_nunits_nparams_mean_ci_max_performance(df):
    n_units_values = np.sort(df["n_units"].unique())
    n_values = len(n_units_values)
    n_trained_parameters_values = np.empty((n_values))
    mean_performances = np.empty((n_values))
    ci_upper_performances = np.empty((n_values))
    ci_lower_performances = np.empty((n_values))
    max_performances = np.empty((n_values))
    for i, n_units in enumerate(n_units_values):
        df_row = df[df["n_units"] == n_units]
        assert len(df_row.index) == 1
        n_trained_parameters_values[i] = df_row["n_trained_parameters"].iloc[0]
        mean_performances[i] = df_row["mean_performance"].iloc[0]
        ci_upper_performances[i] = df_row["ci_upper_performance"].iloc[0]
        ci_lower_performances[i] = df_row["ci_lower_performance"].iloc[0]
        max_performances[i] = df_row["max_performance"].iloc[0]
    return n_units_values, n_trained_parameters_values, mean_performances, ci_upper_performances, ci_lower_performances, max_performances

def nunits_nparams_for_network_to_exceed_nparams(network, nparams_min):
    """
    returns the first (number of units, number of parameters) pair
    for the given network architecture to exceed the given number of parameters
    """
    poly_coeffs = network._get_n_trainable_parameters_polynomial_coefficients_of_n_hunits()
    roots = polynomial.polyroots(polynomial.polysub(poly_coeffs, (nparams_min, )))
    nunits = math.ceil(min(roots[roots > 0]))
    nparams = int(polynomial.polyval(nunits, poly_coeffs))
    return nunits, nparams

def configure_xaxis(ax, xlabel, xticks, xlabel_params, xticks_params, xticklabels_params,
    xlim, xscale="log", twin_offset_pt=34):
    ax.set_xscale(xscale)
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel(xlabel)
    # twin axis to show number of parameters below number of units
    ax_params = ax.twiny()
    ax_params.set_xlim(xlim)
    ax_params.set_xscale(xscale)
    ax_params.set_xticks(xticks_params)
    ax_params.set_xticklabels(xticklabels_params)
    ax_params.set_xlabel(xlabel_params)
    ax_params.spines["bottom"].set_position(("outward", twin_offset_pt))
    ax_params.xaxis.set_label_position("bottom")
    ax_params.xaxis.set_ticks_position("bottom")
    ax_params.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax_params.spines["left"].set_visible(False)


parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="data file path")
parser.add_argument("--second_data_path", default=None, type=str, help="additional data file path")
parser.add_argument("--nparams_indicator", default=475, type=int)
utils.add_arguments(parser, ["output", "style", "legend_style", "width", "height"])
args = parser.parse_args()
data_path = args.data_path
second_data_path = args.second_data_path
out_path = args.output
style = args.style
legend_style = args.legend_style
width_argin = args.width
height_argin = args.height

df = pandas.read_csv(data_path)
n_units_values, n_trained_parameters_values, mean_performances, ci_upper_performances, ci_lower_performances, max_performances = get_nunits_nparams_mean_ci_max_performance(df)
n_values = len(n_units_values)

has_second_data = second_data_path is not None
if has_second_data:
    df_second = pandas.read_csv(second_data_path)
    n_units_values_second, n_trained_parameters_values_second, mean_performances_second, ci_upper_performances_second, ci_lower_performances_second, max_performances_second = get_nunits_nparams_mean_ci_max_performance(df_second)

network_path = df["sample_network_path"].iloc[0]
network = model.Network.load(network_path)
validate_dataset_name = df["validate_dataset_name"].iloc[0]
gen_process = data.load_gen_process(validate_dataset_name)
color = plots.get_bar_color_with_model(network, gen_process, legend_style=legend_style, metric_kind="performance")

nunits_ind, nparams_ind = nunits_nparams_for_network_to_exceed_nparams(network, args.nparams_indicator)

plots.configure_plot_style(style)

should_plot_max_performance = True

figsize_w = 1.72
if width_argin is not None:
    figsize_w = width_argin
figsize_h = 1.72 + 176 / 300
if height_argin is not None:
    figsize_h = height_argin
figsize = (figsize_w, figsize_h)

xlabel = "Num. of units"
xlabel_params = "Num. of trained\nparameters"

if gen_process.is_markov():
    ymin = 50
else:
    ymin = 70
ymax = 100.

fig = plt.figure(figsize=figsize, constrained_layout=True)
if has_second_data:
    # plot second data on a second column
    width_ratios = [n_values, 1]
    gridspec = fig.add_gridspec(nrows=1, ncols=2, width_ratios=width_ratios,
        wspace=0)
    ax = fig.add_subplot(gridspec[0])
    ax_sharey = ax
else:
    ax = fig.gca()
ax.plot(n_units_values, mean_performances,
    color=color, linestyle="-")
ax.fill_between(n_units_values, ci_upper_performances, ci_lower_performances,
    color=color, alpha=plots.ALPHA_CI)
if should_plot_max_performance:
    ax.plot(n_units_values, max_performances,
        color=color, linestyle="--")

xticks = [n_units_values[i] for i in [0, n_values // 2, -1]]
xlim = n_units_values[0], n_units_values[-1]
if nunits_ind <= xlim[1]:
    xticks_params = [n_units_values[0], nunits_ind]
    xticklabels_params = [f"{int(nparams):d}" for nparams in (n_trained_parameters_values[0], nparams_ind)]
else:
    xticks_params = [n_units_values[0]]
    xticklabels_params = [f"{int(n_trained_parameters_values[0]):d}"]

configure_xaxis(ax, xlabel, xticks, xlabel_params, xticks_params, xticklabels_params, xlim)
ax.set_ylim(ymin, ymax)
ax.set_ylabel(plots.AXISLABEL_PERFORMANCE_SHORT)
ax.yaxis.set_major_formatter(plots.get_formatter_percent())
ax.spines["top"].set_visible(True)

if has_second_data:
    n_units_ax2 = np.concatenate(([n_units_values[-1]], n_units_values_second))
    mean_performances_ax2 = np.concatenate(([mean_performances[-1]], mean_performances_second))
    ci_upper_performances_ax2 = np.concatenate(([ci_upper_performances[-1]], ci_upper_performances_second))
    ci_lower_performances_ax2 = np.concatenate(([ci_lower_performances[-1]], ci_lower_performances_second))
    max_performances_ax2 = np.concatenate(([max_performances[-1]], max_performances_second))

    ax2 = fig.add_subplot(gridspec[1], sharey=ax_sharey)
    ax2.plot(n_units_ax2, mean_performances_ax2,
        color=color, linestyle="-")
    ax2.fill_between(n_units_ax2, ci_upper_performances_ax2, ci_lower_performances_ax2,
        color=color, alpha=plots.ALPHA_CI)
    if should_plot_max_performance:
        ax2.plot(n_units_ax2, max_performances_ax2,
            color=color, linestyle="--")
    xticks_ax2 = [n_units_values_second[-1]]
    xmax_ax2 = n_units_values_second[-1]
    xmin_ax2 = min(n_units_values_second[0],  xmax_ax2-1)
    xlim_ax2 = xmin_ax2, xmax_ax2
    xticks_params_ax2 = [nunits_ind]
    xticklabels_params_ax2 = [f"{int(nparams_ind):d}"]
    configure_xaxis(ax2, "", xticks_ax2, "", xticks_params_ax2, xticklabels_params_ax2, xlim_ax2, xscale="linear")
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.spines["top"].set_visible(True)

fig.savefig(out_path)
print("Figure saved at", out_path)
