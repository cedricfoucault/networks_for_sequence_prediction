import argparse
import collections
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
parser.add_argument("--group_ids", nargs="*", help="group ids to plot when available")
parser.add_argument("--data_paths", nargs="*", help="data paths to test on")
utils.add_arguments(parser, ["output", "style", "legend_style", "width", "height"])
args = parser.parse_args()
data_paths = args.data_paths
out_path = args.output
style = args.style
group_ids = args.group_ids
legend_style = args.legend_style
width_argin = args.width
height_argin = args.height

# Retrieve the plot data by agent, training environment, and testing environment
# using a dictionary (the labels are used as keys).
# one groupid corresponding to a pair (agent, training environment).
plotdata_by_grouplabel_trainlabel_testlabel = collections.OrderedDict()
gen_process_labels = []
for data_path in data_paths:
    df = pandas.read_csv(data_path)
    group_ids = utils.list_members_of_other_list(group_ids, df["group_id"].unique())
    test_gen_process = data.load_gen_process(df["dataset_name"].iloc[0])
    test_label = test_gen_process.get_label(abbreviated=True)
    train_labels = [training_data.load_gen_process(group_id).get_label(abbreviated=True) for group_id in group_ids]
    performances, group_labels, sublabels, colors = get_performances_labels_colors(df, group_ids,
        legend_style=legend_style, test_gen_process=test_gen_process)
    for i, performance in enumerate(performances):
        group_label = group_labels[i]
        train_label = train_labels[i]
        plotdata_by_trainlabel_testlabel = plotdata_by_grouplabel_trainlabel_testlabel.setdefault(group_label, {})
        plotdata_by_testlabel = plotdata_by_trainlabel_testlabel.setdefault(train_label, {})
        plotdata_by_testlabel[test_label] = {"performance": performance, "color": colors[i]}
        plotdata_by_trainlabel_testlabel[train_label] = plotdata_by_testlabel
        plotdata_by_grouplabel_trainlabel_testlabel[group_label] = plotdata_by_trainlabel_testlabel
    gen_process_labels.append(test_label)

# Format the plot data as a list, with one item per bar on the bar plot, from top to bottom
group_labels = []
train_labels = []
test_labels = []
performances = []
colors = []
ypos = []
y = 0
for group_label, plotdata_by_trainlabel_testlabel in plotdata_by_grouplabel_trainlabel_testlabel.items():
    for i_train, train_label in enumerate(gen_process_labels):
        for i_test, test_label in enumerate(gen_process_labels):
            group_labels.append(group_label if (i_train == 0 and i_test == 0) else "")
            train_labels.append(train_label if i_test == 0 else "")
            test_labels.append(test_label)
            performances.append(plotdata_by_trainlabel_testlabel[train_label][test_label]["performance"])
            colors.append(plotdata_by_trainlabel_testlabel[train_label][test_label]["color"])
            ypos.append(y)
            y -= 1
        y -= 1
    y -= 1
performances = np.array(performances)
ypos = np.array(ypos)
yposmax, yposmin = ypos.max(), ypos.min()
ymin, ymax = yposmin - 0.5, yposmax + 0.5

# Configure plot, figure, axes
plots.configure_plot_style(style)

figsize_h = 7.92 / 36. * (yposmax - yposmin)
figsize_w = 6.97
xlabel = plots.AXISLABEL_PERFORMANCE
if width_argin is not None:
    figsize_w = width_argin
if height_argin is not None:
    figsize_h = height_argin
figsize = (figsize_w, figsize_h)

fig = plt.figure(figsize=figsize)
left_margin = 3.01 / figsize_w # prop. of figure size
right_margin = 0.27 / figsize_w # prop. of figure size
top_margin = 0.04 / figsize_h # prop. of figure size
bottom_margin = 0.58 / figsize_h # prop. figure size
axes_rect = [left_margin, bottom_margin, 1. - right_margin - left_margin, 1. - top_margin - bottom_margin]
ax = fig.add_axes(axes_rect)

# Plot bar plot
bar_height = 1.
plot_performances_barh(ax, performances, [], [], colors, None,
    xlabel, ypos=ypos, bar_height=bar_height)
ax.set_ylim(ymin, ymax)

# Add label to specify which agent, training environment, and testing environment each bar corresponds to
x_group_label = 0. # inches
x_train_label = 1.3 # inches
x_test_label = 2.17 # inches
y_increment = figsize_h * (1 - top_margin - bottom_margin) / (ymax - ymin) # inches
y_start = figsize_h * (1 - top_margin) - y_increment / 2 # inches
for i, group_label in enumerate(group_labels):
    y = y_start + ypos[i] * y_increment
    train_label = train_labels[i]
    test_label = test_labels[i]
    fig.text(x_group_label, y, group_label, ha="left", va="center", transform=fig.dpi_scale_trans)
    fig.text(x_train_label, y, train_label, ha="left", va="center", transform=fig.dpi_scale_trans)
    fig.text(x_test_label, y, test_label, ha="left", va="center", transform=fig.dpi_scale_trans)

fig.savefig(out_path)
print("Figure saved at", out_path)

