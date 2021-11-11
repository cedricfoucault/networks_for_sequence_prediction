import argparse
import data
import decoding_data
import io_model
import matplotlib.pyplot as plt
import measure
import numpy as np
import os
import plots
import sequence
import training_data
import utils

parser = argparse.ArgumentParser()
parser.add_argument("group_id", type=str, help="group ids to plot")
parser.add_argument("decoder_group_dir", type=str, help="path to decoder group directory")
utils.add_arguments(parser, ["output", "style", "width"])
parser.add_argument("--dataset_name", type=str, default="B_PC1by75_N1000_06-03-20", help="name of test dataset")
parser.add_argument("--i_sequence", type=int, default=2, help="index of the sequence to plot")
parser.add_argument("--n_time_steps", type=int, default=None, help="number of time steps to plot")
parser.add_argument('--hide_ylabel', dest='hide_ylabel', action='store_true')
parser.set_defaults(hide_ylabel=False)
parser.add_argument('--include_io', dest='include_io', action='store_true')
parser.set_defaults(include_io=False)
args = parser.parse_args()
group_id = args.group_id
decoder_group_dir = args.decoder_group_dir
out_path = args.output
style = args.style
width_argin = args.width
dataset_name = args.dataset_name
i_sequence = args.i_sequence
n_time_steps_plot = args.n_time_steps
hide_ylabel = args.hide_ylabel
include_io = args.include_io

# Load network and decoder

predictor_key = "hidden_state"
confidence_key = "io_confidence"

networks, network_ids = training_data.load_models_ids(group_id)
confidence_decoders = decoding_data.load_decoders(decoder_group_dir, confidence_key, predictor_key, network_ids)

median_network_id = decoding_data.get_median_r2_network_id(decoder_group_dir, confidence_key, predictor_key)
i_network = list(network_ids).index(median_network_id)

network = networks[i_network]
decoder = confidence_decoders[i_network]

# Get sequence data on test sequence

test_data = data.load_test_data(dataset_name)
inputs, _, p_gens = test_data

inputs_plot = inputs[:, i_sequence:(i_sequence+1), :]
if n_time_steps_plot is not None:
    inputs_plot = inputs_plot[0:n_time_steps_plot, :, :]

activations = network.forward_activations(inputs_plot)
predictions = activations["outputs"]
confidences = measure.get_decoded_tensor(decoder, activations["hidden_state"])
lrs = measure.get_learning_rate(predictions, inputs_plot)

input_seq = sequence.get_numpy1D_from_tensor(inputs_plot)
prediction_seq = sequence.get_numpy1D_from_tensor(predictions)
confidence_seq = sequence.get_numpy1D_from_tensor(confidences)
lr_seq = sequence.get_numpy1D_from_tensor(lrs)

if include_io:
    gen_process = data.load_gen_process(dataset_name)
    io = io_model.IOModel(gen_process)
    io_outputs = io.get_output_stats(inputs_plot)
    io_confidences = measure.get_confidence_from_sds(io_outputs["sd"])
    io_lrs = measure.get_learning_rate(io_outputs["mean"], inputs_plot)
    io_confidence_seq = sequence.get_numpy1D_from_tensor(io_confidences)
    io_lr_seq = sequence.get_numpy1D_from_tensor(io_lrs)

# Plot the sequence
plots.configure_plot_style(style)

fig_h = 2.16
if width_argin is not None:
    fig_w = width_argin
else:
    fig_w = 4.92
figsize = (fig_w, fig_h)

confidence_label = decoding_data.KEY_LABEL_DICT[confidence_key] + " readout"
lr_label = plots.AXISLABEL_LEARNINGRATE
color_confidence = "#0AAA02"
color_lr = "#EC138E"

fig = plt.figure(figsize=figsize, constrained_layout=True)
ax = fig.gca()
ax_lr = ax.twinx()
ax_obs = ax.twinx()
time_seq = np.arange(len(input_seq))
ax_obs.plot(time_seq, input_seq, 'o', color=plots.COLOR_OBSERVATION, markersize=.5)
ax.plot(time_seq, confidence_seq, '-', color=color_confidence)
ax_lr.plot(time_seq, lr_seq, '-', color=color_lr)
if include_io:
    ax.plot(time_seq, io_confidence_seq, '--', color=color_confidence)
    ax.plot(time_seq, io_lr_seq, '--', color=color_lr)
ax.set_xlabel(plots.AXISLABEL_TIME)
ax.set_xlim(-1, len(time_seq))
ax.spines['right'].set_visible(True)
if not hide_ylabel:
    ax.set_ylabel(confidence_label, color=color_confidence)
    ax_lr.set_ylabel(lr_label, color=color_lr)
ax.tick_params(axis='y', labelcolor=color_confidence)
ax_lr.tick_params(axis='y', labelcolor=color_lr)
ax.yaxis.set_major_formatter(plots.get_formatter_strip_leading_0_with_n_decimals(1))
ax_lr.yaxis.set_major_formatter(plots.get_formatter_strip_leading_0_with_n_decimals(1))
ax_obs.yaxis.set_visible(False)

fig.savefig(out_path)
print("Figure saved at", out_path)
