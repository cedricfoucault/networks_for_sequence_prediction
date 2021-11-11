import argparse
import io_model
import matplotlib.pyplot as plt
import os
import plots
import torch
import training_data
import utils

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="data file path")
parser.add_argument("group_ids", nargs="*", help="group ids to plot when available")
utils.add_arguments(parser, ["output", "style", "legend_style", "width"])
parser.add_argument('--hide_io', dest='hide_io', action='store_true')
parser.set_defaults(hide_io=False)
parser.add_argument('--hide_legend', dest='hide_legend', action='store_true')
parser.set_defaults(hide_legend=False)
parser.add_argument("--n_time_steps", type=int, default=None)
parser.add_argument('--should_annotate_right', dest='should_annotate_right', action='store_true')
parser.set_defaults(should_annotate_right=False)
args = parser.parse_args()
data_path = args.data_path
group_ids = args.group_ids
out_path = args.output
style = args.style
legend_style = args.legend_style
width_argin = args.width
hide_io = args.hide_io
hide_legend = args.hide_legend
should_annotate_right = args.should_annotate_right

data_dict = torch.load(data_path)
n_time_steps = data_dict["n_time_steps"]
reversal_times = data_dict["reversal_times"]
data_io = data_dict["data_io"]
data_by_groupid = data_dict["data_by_groupid"]

# Plot
plots.configure_plot_style(style)
width = 4.92
height = 2.16
linewidth = float(plt.rcParams["lines.linewidth"])
labelfontsize = plt.rcParams["legend.fontsize"]
if style.lower() == "presentation":
    width = 9.45
    height = 4.2
    legend_loc = 'upper left'
if width_argin is not None:
    width = width_argin
n_time_steps_displayed = args.n_time_steps if args.n_time_steps is not None else n_time_steps

figsize = (width, height)
fig = plt.figure(figsize=figsize, constrained_layout=True)
ax = fig.gca()
time_steps = range(n_time_steps)

def plot_with_data(time_steps, data, label, color):
    plt.plot(data["mean"], '-', linewidth=linewidth, label=label, color=color)
    if ("ci_lower" in data) and ("ci_upper" in data):
        ci_lower = data["ci_lower"]
        ci_upper = data["ci_upper"]
        plt.fill_between(time_steps, ci_lower, ci_upper, alpha=plots.ALPHA_CI, color=color)

if not hide_io:
    plot_with_data(time_steps, data_io, io_model.IOModel.get_class_label(), plots.COLOR_LINE_IO)
for group_id in group_ids:
    if group_id not in data_by_groupid:
        continue
    label = plots.get_label_with_group_id(group_id, legend_style=legend_style)
    color = plots.get_line_color_with_group_id(group_id, legend_style=legend_style)
    plot_with_data(time_steps, data_by_groupid[group_id], label, color)

reversal_times_displayed = reversal_times[reversal_times < n_time_steps_displayed - 1]
for t in reversal_times_displayed:
    plt.axvline(x=t, linestyle="-", linewidth=linewidth, color="black")
plt.xlabel(plots.AXISLABEL_TIME)
plt.xlim(0, n_time_steps_displayed)
plt.ylabel(plots.AXISLABEL_LEARNINGRATE)
# "change points" text and arrows
# ymin, ymax = plt.ylim()
ymin, ymax = 0.05, 0.275
plt.ylim(ymin, ymax)
last_reversal_time_displayed = reversal_times_displayed[-1]
if not hide_legend:
    x_legend_anchor = (last_reversal_time_displayed + 15) / (n_time_steps_displayed)
    y_legend_anchor = (0.22 - ymin) / (ymax - ymin)#(change_points_text_y - ymin) / (ymax - ymin)
    plt.legend(loc='upper left', bbox_to_anchor=(x_legend_anchor, y_legend_anchor),
               frameon=False, borderpad=0, borderaxespad=0)
change_points_text = plots.TEXTLABEL_CHANGEPOINT
change_points_text_x = reversal_times[0] // 2
change_points_text_ha = "center"
change_points_text_y = ymax
change_points_arrow_x = reversal_times[0]
change_points_arrow_y = ymax * 0.9
if should_annotate_right:
    change_points_arrow_x = last_reversal_time_displayed
    change_points_text_x = n_time_steps_displayed - (n_time_steps_displayed - last_reversal_time_displayed - 1) // 2
    change_points_text_ha = "left"
plt.annotate(change_points_text,
             (change_points_arrow_x, change_points_arrow_y),
             (change_points_text_x, change_points_text_y),
             horizontalalignment=change_points_text_ha,
             verticalalignment="top",
             arrowprops = dict(arrowstyle="->", shrinkA=0., shrinkB=0., linewidth=linewidth),
             color="black",
             fontsize=labelfontsize)

fig.savefig(out_path)


