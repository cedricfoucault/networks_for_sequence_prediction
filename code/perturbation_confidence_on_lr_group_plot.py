import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plots
from scipy import stats
import utils

# Plot mean delta_lr of each network as a function of delta_conf

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="perturbation experiment data path")
utils.add_arguments(parser, ["output", "style", "width", "height"])
args = parser.parse_args()
data_path = args.data_path
out_path = args.output
style = args.style
width_argin = args.width
height_argin = args.height

data = np.load(data_path)
delta_confs = data["delta_confs"]
delta_lrs = data["delta_lrs"]
n_delta_confs = delta_lrs.shape[0]
n_networks = delta_lrs.shape[1]
assert n_delta_confs == delta_confs.shape[0]
delta_conf_step = delta_confs[1] - delta_confs[0]
delta_lr_means = delta_lrs.mean(axis=2)

ymin = min(-0.018, delta_lr_means.min() * 1.01)
ymax = max(+0.018, delta_lr_means.max() * 1.01)
xmin, xmax = delta_confs[0]-delta_conf_step/4, delta_confs[-1]+delta_conf_step/4

plots.configure_plot_style(style)

figsize_w = 3.30
figsize_h = 2.71
stat_fontsize = plt.rcParams['legend.fontsize']
xlabel = "Induced change in precision"
ylabel = "Effect on subsequent learning rate"
if style.lower() == "presentation":
    figsize_w = 4.71
    figsize_h = 3.54
if width_argin is not None:
    figsize_w = width_argin
if height_argin is not None:
    figsize_h = height_argin
    if height_argin < 1.35:
        xlabel = ""
        ylabel = ""

figsize = (figsize_w, figsize_h)

fig = plt.figure(figsize=figsize, constrained_layout=True)
ax = fig.gca()
dataset = [delta_lr_means[i_conf, :] for i_conf in range(n_delta_confs)]
plt.violinplot(dataset,
                positions=delta_confs,
                widths=delta_conf_step/2,
                showextrema=False)
for i_network in range(n_networks):
    # jitter = np.random.uniform(-delta_conf_step/4, +delta_conf_step/4, n_delta_confs)
    jitter = 0
    plt.plot(delta_confs + jitter, delta_lr_means[:, i_network], '.-', color=plots.COLOR_DOT_SAMPLE, markersize=2, linewidth=0.5)
dataset = [delta_lr_means[i_conf, :] for i_conf in range(n_delta_confs)]
    
plt.axvline(x=0, linewidth=1, color="#000000", ls='--', lw=.5, alpha=.5)
plt.axhline(y=0, linewidth=1, color="#000000", ls='--', lw=.5, alpha=.5)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
ax.xaxis.set_major_formatter(plots.get_formatter_strip_leading_0_with_n_decimals(2))
ax.yaxis.set_major_formatter(plots.get_formatter_strip_leading_0_with_n_decimals(2))

n_xticks = min(11, n_delta_confs)
xticks = np.linspace(delta_confs[0], delta_confs[-1], num=n_xticks)
plt.xticks(xticks)

# compute and draw t test stats
stat_color = "black"
stat_y = min(delta_lr_means.max() * 1.05, 0.97 * ymax)
stat_y_pad = delta_lr_means.max() * 0.02
stat_x_margin = delta_conf_step / 20
for i_conf in range(n_delta_confs - 1):
    t_val, p_val = stats.ttest_rel(delta_lr_means[i_conf, :], delta_lr_means[i_conf+1, :])
    p_val_onetailed = p_val / 2 if t_val >= 0. else 1
    text = utils.stat_label(p_val_onetailed)
    x1 = delta_confs[i_conf] + stat_x_margin / 2
    x2 = delta_confs[i_conf+1] - stat_x_margin / 2
    plt.plot([x1, x1, x2, x2], [stat_y, stat_y+stat_y_pad, stat_y+stat_y_pad, stat_y],
             linewidth=0.5, color=stat_color)
    plt.text((x1 + x2) / 2, stat_y+stat_y_pad, text, ha='center', va='bottom',
             color=stat_color, fontsize=stat_fontsize)

fig.savefig(out_path)
print("Figure saved at", out_path)
