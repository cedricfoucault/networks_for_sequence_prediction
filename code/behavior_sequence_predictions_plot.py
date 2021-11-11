import argparse
import data
import io_model
import matplotlib.pyplot as plt
import numpy as np
import os
import performance_data
import plots
import sequence
import torch
import training_data
import utils

parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str, help="name of test dataset")
parser.add_argument("i_sequence", type=int, help="index of the sequence to plot")
parser.add_argument("group_ids", nargs="*", help="group ids to plot")
parser.add_argument("performance_stats_path", type=str,
    help="path to CSV file that contains index of models with median performance")
utils.add_arguments(parser, ["output", "style", "legend_style", "width"])
args = parser.parse_args()
dataset_name = args.dataset_name
i_sequence = args.i_sequence
group_ids = args.group_ids
performance_stats_path = args.performance_stats_path
out_path = args.output
style = args.style
legend_style = args.legend_style
width_argin = args.width

# Get test data
test_data = data.load_test_data(dataset_name)
inputs, _, p_gens = test_data
gen_process = data.load_gen_process(dataset_name)
io = io_model.IOModel(gen_process)

input_seq = sequence.sample_from_tensor(inputs, i_sequence)
inputs_imax = inputs[:, i_sequence:(i_sequence+1), :]
io_predictions_imax =  io.get_predictions(inputs_imax)
io_prediction_seq = io_predictions_imax.flatten().detach().numpy()

# Plot the sequence
plots.configure_plot_style(style)

width = 4.92
height = 2.16
legend_loc = 'upper right'
legend_bbox_to_anchor = (1., 0.9)
if style.lower() == "presentation":
    width = 9.37
    height = 4.2
    legend_loc = 'upper left'
    legend_bbox_to_anchor = (0., 0.95)
if width_argin is not None:
    width = width_argin

figsize = (width, height)
fig = plt.figure(figsize=figsize, constrained_layout=True)
ax = fig.gca()
io_label = io.get_label()
time_seq = np.arange(len(input_seq))
plt.plot(time_seq, input_seq, 'o', color=plots.COLOR_OBSERVATION, markersize=plots.MARKERSIZE_OBSERVATION)
plt.plot(time_seq, io_prediction_seq, '-', label=io_label, color=plots.COLOR_LINE_IO)
for group_id in group_ids:
    model_i = performance_data.get_median_performance_model(group_id, performance_stats_path)
    model_predictions_imax = model_i(inputs_imax)
    prediction_seq = model_predictions_imax.flatten().detach().numpy()
    label = plots.get_label_with_group_id(group_id, legend_style=legend_style)
    color = plots.get_line_color_with_group_id(group_id, legend_style=legend_style)
    plt.plot(time_seq, prediction_seq, '-', label=label, color=color)
plt.xlabel(plots.AXISLABEL_TIME)
plt.ylabel(plots.AXISLABEL_PREDICTION)
plt.xlim(-1, len(time_seq))
plt.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor,
           frameon=False)

fig.savefig(out_path)
print("Figure saved at", out_path)
