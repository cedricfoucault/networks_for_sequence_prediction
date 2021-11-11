import argparse
import data
import matplotlib as mpl
import matplotlib.pyplot as plt
import measure
import model
import numpy as np
import os
import pandas
from performance_test_plot_common import get_performances_labels_colors, plot_performances_barh
import plots
import training_data
import utils

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="individual data path")
parser.add_argument("group_ids", nargs="*", help="group ids to plot when available")
utils.add_arguments(parser, ["output", "style", "legend_style", "width", "height"])
args = parser.parse_args()
data_path = args.data_path
out_path = args.output
style = args.style
group_ids = args.group_ids
legend_style = args.legend_style
width_argin = args.width
height_argin = args.height

df = pandas.read_csv(data_path)
group_ids = utils.list_members_of_other_list(group_ids, df["group_id"].unique())
n_groups = len(group_ids)
test_gen_process = data.load_gen_process(df["dataset_name"].iloc[0])

performances, labels, sublabels, colors = get_performances_labels_colors(df, group_ids,
    legend_style=legend_style, test_gen_process=test_gen_process)

plots.configure_plot_style(style)

figsize_h_inbarzone_per_group = 114 / 300
figsize_h_outbarzone = (810/300 - 5*figsize_h_inbarzone_per_group)
figsize_h = figsize_h_outbarzone + figsize_h_inbarzone_per_group * n_groups
figsize_w = 6.56
bar_fontsize = mpl.rcParams['axes.labelsize']
xlabel = plots.AXISLABEL_PERFORMANCE
xticks = range(0, 101, 10)
if style.lower() == "presentation":
    figsize_w = 10.11
    figsize_h = 3.78
    bar_fontsize = 18
if width_argin is not None:
    if width_argin < figsize_w:
        xticks = range(0, 101, 50)
    figsize_w = width_argin
if height_argin is not None:
    figsize_h = height_argin

figsize = (figsize_w, figsize_h)

fig = plt.figure(figsize=figsize, constrained_layout=True)
fig.set_constrained_layout_pads(h_pad=0.1)
ax = fig.gca()            

plot_performances_barh(ax, performances, labels, sublabels, colors, bar_fontsize,
    xlabel, xticks=xticks)

fig.savefig(out_path)
print("Figure saved at", out_path)

