import argparse
from decoding_bernoulli_plot_common import plot_dots_per_group, get_formatter_optimal_value
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas
import plots
import utils

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="path to raw data file")
parser.add_argument("group_ids", nargs="*", help="group ids to plot when available")
utils.add_arguments(parser, ["output", "style", "legend_style", "width", "height"])
args = parser.parse_args()
data_path = args.data_path
group_ids = args.group_ids
out_path = args.output
style = args.style
legend_style = args.legend_style
width_argin = args.width
height_argin = args.height

df = pandas.read_csv(data_path)

dataset_name = df["dataset_name"].iloc[0]
io_pearsonr = df["io_pearsonr"].iloc[0]
predictor_key = "hidden_state"

n_groups = len(group_ids)
y_groups = np.arange(n_groups)
pearsonrs_per_group = []
pearsonr_median_per_group = []
labels = []
# min_pearsonr = io_pearsonr
# max_pearsonr = io_pearsonr
for group_id in group_ids:
    label = plots.get_label_with_group_id(group_id, legend_style=legend_style)
    dfg = df.query("(group_id == @group_id)")
    dfgp = dfg.query("(predictor_key == @predictor_key)")
    pearsonrs = dfgp["pearsonr"].to_numpy()
    pearsonrs_per_group.append(pearsonrs)
    pearsonr_median_per_group.append(np.median(pearsonrs))
    labels.append(label)
    # min_pearsonr = min(min_pearsonr, pearsonrs.min())
    # max_pearsonr = max(max_pearsonr, pearsonrs.max())
# print("io_pearsonr", io_pearsonr)
# print("min_pearsonr", min_pearsonr)
# print("max_pearsonr", max_pearsonr)

plots.configure_plot_style(style)

if width_argin is not None:
    fig_w = width_argin
else:
    fig_w = 2.01
if height_argin is not None:
    fig_h = figh_argin
else:
    fig_h = 2.01
figsize = (fig_w, fig_h)

fig = plt.figure(figsize=figsize, constrained_layout=True)
ax = fig.gca()
ax.invert_xaxis()
ax.invert_yaxis()
plot_dots_per_group(ax, pearsonrs_per_group, pearsonr_median_per_group, labels)
ax.set_xticks([0, io_pearsonr / 2, io_pearsonr])
ax.xaxis.set_major_formatter(get_formatter_optimal_value(io_pearsonr))
ax.set_title("Correlation with\nlearning rate")

fig.savefig(out_path)
print("Figure saved at", out_path)

