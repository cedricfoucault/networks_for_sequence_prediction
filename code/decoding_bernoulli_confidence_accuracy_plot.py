import argparse
from decoding_bernoulli_plot_common import plot_dots_per_group, get_formatter_optimal_value
import decoding_data
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas
import plots
import utils

parser = argparse.ArgumentParser()
parser.add_argument("decoder_group_dirs", nargs='*', help="path to decoder group directories")
parser.add_argument("--group_ids", nargs="*", help="group ids to plot")
utils.add_arguments(parser, ["output", "style", "legend_style", "width", "height"])
args = parser.parse_args()
decoder_group_dirs = args.decoder_group_dirs
group_ids = args.group_ids
out_path = args.output
style = args.style
legend_style = args.legend_style
width_argin = args.width
height_argin = args.height

predictor_key = "hidden_state"
outcome_key = "io_confidence"
io_pearsonr = 1.

n_groups = len(group_ids)
y_groups = np.arange(n_groups)
labels = []
pearsonrs_per_group = []
pearsonr_median_per_group = []
for i, group_id in enumerate(group_ids):
    label = plots.get_label_with_group_id(group_id, legend_style=legend_style)
    decoder_group_dir = decoder_group_dirs[i]
    regression_data_path = decoding_data.get_group_regression_data_path(decoder_group_dir)
    df_individual = pandas.read_csv(regression_data_path)
    rows_individual = df_individual.query("outcome == @outcome_key & predictor == @predictor_key")
    pearsonrs = rows_individual["r"].to_numpy()
    pearsonr_median = np.median(pearsonrs)
    pearsonrs_per_group.append(pearsonrs)
    pearsonr_median_per_group.append(pearsonr_median)
    labels.append(label)

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
ax.invert_yaxis()
plot_dots_per_group(ax, pearsonrs_per_group, pearsonr_median_per_group, labels)
ax.set_xticks([0, io_pearsonr / 2, io_pearsonr])
ax.set_xlim(0, io_pearsonr)
ax.xaxis.set_major_formatter(get_formatter_optimal_value(io_pearsonr))

ax.set_title("Accuracy of the\nread precision")

fig.savefig(out_path)
print("Figure saved at", out_path)


