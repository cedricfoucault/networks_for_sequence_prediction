import argparse
from dynamics_common import (T_PLOT_START, COLOR_BY_LEVEL, DEFAULT_SEED,
    get_inputs_by_level_with_seed, print_inputs_by_level)
import generate as gen
import io_model
import itertools
import matplotlib.pyplot as plt
import numpy as np
import plots
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
utils.add_arguments(parser, ["output", "width", "height"])
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--output_singletime", type=str, default=None)
args = parser.parse_args()
out_path = args.output

inputs_by_level = get_inputs_by_level_with_seed(args.seed)
if args.verbose:
    print_inputs_by_level(inputs_by_level)

figsize = args.width, args.height
plots.configure_plot_style("paper")

# compute ideal observer posterior
resol = 101
gen_process = gen.GenerativeProcessBernoulliRandom(1 / 75.)
io = io_model.IOModel(gen_process)
io_outputs_by_level = [ io.get_outputs(inputs, resol=resol) for inputs in inputs_by_level ]
io_dist_by_level = [ io_outputs["dist"].squeeze().detach().numpy() for io_outputs in io_outputs_by_level ]
io_means_by_level = [ io_outputs["mean"].squeeze().detach().numpy() for io_outputs in io_outputs_by_level ]

# find times in each level within the start of the streak
# where the predictions are the closest to each other across levels
min_mean_distance = 1
t_io_equal_mean_by_level = None
n_levels = len(inputs_by_level)
for t_by_level in itertools.product(range(T_PLOT_START, T_PLOT_START + 6), repeat=n_levels):
    means = np.array([io_means_by_level[i_level][t] for i_level, t in enumerate(t_by_level)])
    mean_distance = means.max() - means.min()
    if mean_distance < min_mean_distance:
        min_mean_distance = mean_distance
        t_io_equal_mean_by_level = np.array(t_by_level)
assert (t_io_equal_mean_by_level is not None), "failed to compute t_io_equal_mean_by_level"
if args.verbose:
    print("t_io_equal_mean_by_level - T_PLOT_START", t_io_equal_mean_by_level - T_PLOT_START)

# plot posterior density across time as a color map, for each level
fig, axes = plt.subplots(figsize=figsize, constrained_layout=True, nrows=n_levels, ncols=1)
vmin, vmax = 0, max([ dist.max() for dist in io_dist_by_level ])
lw_vline = 1.
for i_level, dist in enumerate(io_dist_by_level):
    ax = axes[i_level]
    image = np.swapaxes(dist[T_PLOT_START:, :], 0, 1)
    ax.imshow(image, origin='upper', vmin=vmin, vmax=vmax, aspect="auto", cmap=plt.cm.viridis)
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.set_yticks([-0.5, resol - 0.5])
    ax.set_yticklabels([1, 0])
    ax.axvline(x=-0.5, ls='-', lw=lw_vline, color=COLOR_BY_LEVEL[i_level])
    ax.axvline(x=t_io_equal_mean_by_level[i_level] - T_PLOT_START-0.5, ls='-', lw=lw_vline, color=COLOR_BY_LEVEL[i_level])
    for edge in ['left', 'right', 'top', 'bottom']:
        ax.spines[edge].set_visible(False)

fig.savefig(out_path)
print("Figure saved at", out_path)

# plot posterior density for one single time point as a line plot,
# for two time points: before the streak, and at the time point when the predictions are equal as found above,
# for each level
output_singletime = args.output_singletime
if output_singletime is not None:
    fig, axes = plt.subplots(figsize=figsize, constrained_layout=True, nrows=n_levels, ncols=2)
    xvalues = np.linspace(start=1., stop=0., num=resol)
    ymax = 0
    for i_level, dist in enumerate(io_dist_by_level):
        for i_col, t in enumerate([T_PLOT_START, t_io_equal_mean_by_level[i_level]]):
            ax = axes[i_level, i_col]
            ax.plot(xvalues, dist[t, :], '-', color=COLOR_BY_LEVEL[i_level])
            mean = np.sum(dist[t, :] * xvalues)
            ax.axvline(x=mean, ls='--', color=COLOR_BY_LEVEL[i_level])
            ax.set_xticks([0, 1])
            ax.set_xlim(0, 1)
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            ymax = max(np.max(ax.get_ylim()[1]), ymax)

    fig.savefig(output_singletime)
    print("Figure saved at", output_singletime)

