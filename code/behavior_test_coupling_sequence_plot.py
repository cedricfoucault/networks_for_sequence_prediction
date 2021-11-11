import argparse
from behavior_test_coupling_gen import create_initial_sequence, _fill_with_repetitions_and_queries
import io_model
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import performance_data
import plots
import sequence
import training_data
import torch
import utils

seed = 29
utils.set_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("group_id_coupled", type=str, help="group id of coupled network models")
parser.add_argument("group_id_independent", type=str, help="group id of independent network models")
parser.add_argument("performance_stats_path_coupled", type=str)
parser.add_argument("performance_stats_path_independent", type=str)
utils.add_arguments(parser, ["output", "style", "width", "height"])
args = parser.parse_args()
group_id_coupled = args.group_id_coupled
group_id_independent = args.group_id_independent
out_path = args.output

network_coupled = performance_data.get_median_performance_model(
    group_id_coupled, args.performance_stats_path_coupled)
network_independent = performance_data.get_median_performance_model(
    group_id_independent, args.performance_stats_path_independent)

gen_process_coupled = training_data.load_gen_process(group_id_coupled)
gen_process_independent = training_data.load_gen_process(group_id_independent)
io_coupled = io_model.IOModel(gen_process_coupled)
io_independent = io_model.IOModel(gen_process_independent)

network_by_coupled = {
    False: network_independent,
    True: network_coupled
}
io_by_coupled = {
    False: io_independent,
    True: io_coupled
}

low_probability = 0.2
rep_item = 1 # the item that will be repeated
other_item = 1 - rep_item
pre_rep_pgen = low_probability
pre_other_pgen = 1 - low_probability
p00 = pre_rep_pgen if rep_item == 0 else pre_other_pgen
p11 = pre_rep_pgen if rep_item == 1 else pre_other_pgen
t_query_before = 74
rep_length = 15
t_query_after = t_query_before + 1 + rep_length

inputs = torch.zeros((t_query_after + 1, 1, 1))
input_initial_sequence = create_initial_sequence(rep_item, pre_rep_pgen, pre_other_pgen, t_query_before)
inputs[0 : t_query_before, 0, 0] = input_initial_sequence
_fill_with_repetitions_and_queries(inputs, rep_item, t_query_before, t_query_after)
input_seq = sequence.get_numpy1D_from_tensor(inputs)

time_seq = np.arange(len(input_seq))

i_start_plot = t_query_before - 16
i_query_before = t_query_before
i_query_after = t_query_after

plots.configure_plot_style(args.style)

figsize = args.width, args.height
fig = plt.figure(figsize=figsize, constrained_layout=True)

ax = fig.gca()
ax.plot(time_seq[i_start_plot:], input_seq[i_start_plot:], 'o',
    color=plots.COLOR_OBSERVATION, markersize=plots.MARKERSIZE_OBSERVATION)
for is_coupled in [True, False]:
    network_prediction_seq = sequence.get_numpy1D_from_tensor(
        network_by_coupled[is_coupled](inputs)
    )
    io_prediction_seq = sequence.get_numpy1D_from_tensor(
        io_by_coupled[is_coupled](inputs)
    )
    color = plots.COLOR_LINE_COUPLED if is_coupled else plots.COLOR_LINE_INDEPENDENT

    ax.plot(time_seq[i_query_before:], network_prediction_seq[i_query_before:], '-',
            color=color)
    ax.plot(time_seq[i_query_before:], io_prediction_seq[i_query_before:], ':',
            color=color)

ax.set_xticks([time_seq[i_start_plot], time_seq[i_query_before], time_seq[i_query_after]])
ax.set_xlabel(plots.AXISLABEL_TIME)
ax.set_ylabel(plots.AXISLABEL_PREDICTION)

fig.savefig(out_path)
print("Figure saved at", out_path)


