import argparse
import data
import io_model
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
import performance_data
import plots
import sequence
import torch
import training_data
import utils

parser = argparse.ArgumentParser()
parser.add_argument("group_ids", nargs="+", help="group ids to plot")
parser.add_argument("performance_stats_path", type=str,
    help="path to CSV file that contains index of models with median performance")
utils.add_arguments(parser, ["output", "style", "legend_style"])
args = parser.parse_args()
group_ids = args.group_ids
performance_stats_path = args.performance_stats_path
out_path = args.output
style = args.style
legend_style = args.legend_style

gen_process = training_data.load_gen_process(group_ids[0])
io = io_model.IOModel(gen_process)

input_seq = np.array([0,0,0,0,1,1,1,1,0,1,0,1], dtype=np.float32)
inputs = torch.tensor(input_seq).reshape(-1, 1, 1)
io_predictions = io.get_predictions(inputs)
io_prediction_seq = sequence.get_numpy1D_from_tensor(io_predictions)

# Plot the sequence
plots.configure_plot_style(style)

fig_w = 1.73
fig_h = 1.40
figsize = (fig_w, fig_h)

io_label = io.get_label()

fig = plt.figure(figsize=figsize, constrained_layout=True)
ax = fig.gca()

def plot_predictions(ax, time_seq, prediction_seq, label, color, marker):
    ax.plot(time_seq, prediction_seq, ls='-', label=label, color=color,
        marker=marker, markersize=plots.MARKERSIZE_PREDICTION)

n_time_steps = input_seq.shape[0]
time_seq = np.arange(n_time_steps)
ax.plot(time_seq, input_seq, 'o', color=plots.COLOR_OBSERVATION, markersize=plots.MARKERSIZE_OBSERVATION)
plot_predictions(ax, time_seq, io_prediction_seq, label=io_label, color=plots.COLOR_LINE_IO, marker=None)
for group_id in group_ids:
    model_i = performance_data.get_median_performance_model(group_id, performance_stats_path)
    model_predictions = model_i(inputs)
    model_prediction_seq = sequence.get_numpy1D_from_tensor(model_predictions)
    model_color = plots.get_line_color_with_group_id(group_id, legend_style=legend_style)
    model_label = plots.get_label_with_group_id(group_id, legend_style=legend_style)
    model_marker = plots.get_marker_with_group_id(group_id, legend_style=legend_style)
    plot_predictions(ax, time_seq, model_prediction_seq, label=model_label, color=model_color, marker=model_marker)
ax.set_xlim(-0.5, n_time_steps)
ax.set_ylabel(plots.AXISLABEL_PREDICTION)
ax.set_xlabel(plots.AXISLABEL_TIME)

fig.savefig(out_path)
print("Figure saved at", out_path)
