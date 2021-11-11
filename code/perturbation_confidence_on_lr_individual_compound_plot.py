import argparse
import os
import matplotlib.pyplot as plt
import measure
import numpy as np
import plots
from scipy import stats
import utils

# Plot delta_lr as a function of delta_conf individually for each network
# on a compound plot showing all network
# and a separate plot showing the network that yields median effect, in bigger size

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="perturbation experiment data path")
parser.add_argument("-o", "--output", help="path to output file for compound plot of individual")
parser.add_argument("-o_median", "--output_median", help="path to output file for plor of median individual")
args = parser.parse_args()
data_path = args.data_path
out_path = args.output
out_path_median = args.output_median

data = np.load(data_path)
delta_confs = data["delta_confs"]
delta_lrs = data["delta_lrs"]
n_delta_confs = delta_lrs.shape[0]
n_networks = delta_lrs.shape[1]
assert n_delta_confs == delta_confs.shape[0]
delta_conf_step = delta_confs[1] - delta_confs[0]

# compute t test p values
p_values = np.empty((n_delta_confs - 1, n_networks))
for i_network in range(n_networks):
    for i_conf in range(n_delta_confs - 1):
        t_val, p_val = stats.ttest_rel(delta_lrs[i_conf, i_network, :], delta_lrs[i_conf+1, i_network, :])
        p_val_onetailed = p_val / 2
        p_values[i_conf, i_network] = p_val_onetailed

delta_lr_stats = measure.get_summary_stats(delta_lrs, axis=2)
# get sorted indices from smallest to greatest lr difference
delta_lr_mean = delta_lr_stats["mean"]
delta_lr_mean_diff = delta_lr_mean[0, :] - delta_lr_mean[-1, :]
indices = np.argsort(delta_lr_mean_diff)

# Plot all networks as small multiple

width_individual = 0.82
aspect_ratio = 4/3
height_individual = width_individual / aspect_ratio
n_cols = 5
n_rows = n_networks // n_cols
assert n_rows * n_cols == n_networks
figsize = (width_individual * n_cols, height_individual * n_rows)

ymin = -0.023
ymax = +0.023
xmin, xmax = delta_confs[0]-delta_conf_step/4, delta_confs[-1]+delta_conf_step/4

plots.configure_plot_style()

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey=True, figsize=figsize)
axes_list = [item for sublist in axes for item in sublist]
plt.subplots_adjust(hspace=0, wspace=0, left=0, bottom=.001, right=.999, top=1)

dot_color = "black"
ecolor = (0, 0, 0, 0.67)

for i_ax, i_network in enumerate(indices):
    ax = axes_list[i_ax]
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(
        which='both',
        bottom=False,
        left=False,
        right=False,
        top=False,
        labelbottom=False,
        labelleft=False
    )
    ax.errorbar(delta_confs, delta_lr_mean[:, i_network],
                 yerr=[delta_lr_mean[:, i_network] - delta_lr_stats["ci_lower"][:, i_network],
                       delta_lr_stats["ci_upper"][:, i_network] - delta_lr_mean[:, i_network]],
                 color=dot_color, ecolor=ecolor, elinewidth=1,
                 fmt='.-',
                 lw=.5, markersize=2)
    # draw t test stats
    stat_color = "black"
    stat_y = ymax * 0.9
    stat_y_pad = ymax * 0.02
    stat_fontsize = 6
    stat_x_margin = delta_conf_step / 20
    for i_conf in range(n_delta_confs - 1):
        x1 = delta_confs[i_conf] + stat_x_margin / 2
        x2 = delta_confs[i_conf+1] - stat_x_margin / 2
        text = utils.stat_label(p_values[i_conf, i_network])
        # ax.plot([x1, x1, x2, x2], [stat_y, stat_y+stat_y_pad, stat_y+stat_y_pad, stat_y],
        #           linewidth=0.5, color=stat_color)
        ax.text((x1 + x2) / 2, stat_y+stat_y_pad, text, ha='center', va='top',
                  color=stat_color, fontsize=stat_fontsize)

fig.savefig(out_path)
print("Figure saved at", out_path)


# Plot the median network in bigger size
# plt.subplots_adjust(hspace=None, wspace=None, left=None, bottom=None, right=None, top=None)

width = 2.22
# width = width_individual * 3
height = width / aspect_ratio
figsize = (width, height)

i_median = indices[n_networks // 2]

fig = plt.figure(figsize=figsize)
ax = plt.gca()
ax.errorbar(delta_confs, delta_lr_mean[:, i_median],
                 yerr=[delta_lr_mean[:, i_median] - delta_lr_stats["ci_lower"][:, i_median],
                       delta_lr_stats["ci_upper"][:, i_median] - delta_lr_mean[:, i_median]],
                 color=dot_color, ecolor=ecolor, elinewidth=1,
                 fmt='.-',
                 lw=.5, markersize=2)
xticks = np.linspace(delta_confs[0], delta_confs[-1], num=n_delta_confs)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xticks(xticks)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.set_xlabel(r"$\Delta c^t$", fontsize=9) # r"change of confidence at time t"
ax.set_ylabel(r"$\Delta \alpha_{eq}^{t+1}$", fontsize=9) # r"change in learning rate at t+1"
# draw t test stats
stat_y = ymax * 0.85
stat_y_pad = ymax * 0.02
stat_fontsize = 9
stat_x_margin = delta_conf_step / 20
for i_conf in range(n_delta_confs - 1):
    x1 = delta_confs[i_conf] + stat_x_margin / 2
    x2 = delta_confs[i_conf+1] - stat_x_margin / 2
    text = utils.stat_label(p_values[i_conf, i_network])
    ax.plot([x1, x1, x2, x2], [stat_y, stat_y+stat_y_pad, stat_y+stat_y_pad, stat_y],
              linewidth=0.5, color=stat_color)
    ax.text((x1 + x2) / 2, stat_y+stat_y_pad, text, ha='center', va='center',
              color=stat_color, fontsize=stat_fontsize)
ax.xaxis.set_major_formatter(plots.get_formatter_strip_leading_0_with_n_decimals(2))
ax.yaxis.set_major_formatter(plots.get_formatter_strip_leading_0_with_n_decimals(2))

fig.subplots_adjust(bottom=0.25, left=0.25, top=0.99, right=0.99, wspace=0, hspace=0)
fig.savefig(out_path_median)
print("Figure saved at", out_path_median)
