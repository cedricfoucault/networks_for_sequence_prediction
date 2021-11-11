import argparse
import data
import matplotlib as mpl
import matplotlib.pyplot as plt
import measure
import numpy as np
import os
import plots
from scipy import stats
import training_data
import utils

# Parse input
parser = argparse.ArgumentParser()
parser.add_argument("group_id", type=str, help="training data group id")
utils.add_arguments(parser, ["output", "style", "width", "height"])
args = parser.parse_args()
group_id = args.group_id
out_path = args.output
style = args.style
width_argin = args.width
height_argin = args.height

# Extract data
network_ids = training_data.load_model_ids(group_id)
n_networks = len(network_ids)
n_updates_s = None
for iNetwork, network_id in enumerate(network_ids):
    progress_dataframe = training_data.load_progress_dataframe(group_id, network_id)
    if n_updates_s is None:
        n_updates_s = progress_dataframe["n_training_updates"].to_numpy()
        loss_progress_s = np.empty((n_networks, len(n_updates_s)))
    else:
        assert np.array_equal(n_updates_s, progress_dataframe["n_training_updates"].to_numpy()), \
        f"Inconsistent n_training_updates: {n_updates_s} vs {progress_dataframe['n_training_updates'].to_numpy()}"
    loss_progress_s[iNetwork, :] = progress_dataframe["validation_loss"].to_numpy()

# Compute performance
metadata_df = training_data.load_metadata(group_id)
validate_dataset_name = metadata_df["config/validate_dataset_name"].iloc[0]
validate_inputs, validate_targets, _ = data.load_test_data(validate_dataset_name)
validate_gen_process = data.load_gen_process(validate_dataset_name)
chance_loss = measure.get_chance_loss(validate_inputs, validate_targets)
io_loss = measure.get_io_loss(validate_inputs, validate_targets, validate_gen_process)
performance_progress_s = measure.get_percent_value(loss_progress_s, chance_loss, io_loss)

# Compute stats
mean_performance_progress = performance_progress_s.mean(axis=0)

def get_select_index(threshold, values, min_ref, max_ref):
    select_value = min_ref + (max_ref - min_ref) * threshold
    return np.argwhere(values >= select_value).min()

# Compute first number of training updates to reach 99% of maximum performance
select_threshold_1 = 0.95
select_threshold_2 = 0.99
min_performance = mean_performance_progress.min()
max_performance = mean_performance_progress.max()
select_index_1 = get_select_index(select_threshold_1, mean_performance_progress, min_performance, max_performance)
select_index_2 = get_select_index(select_threshold_2, mean_performance_progress, min_performance, max_performance)
select_n_updates_1 = n_updates_s[select_index_1]
select_n_updates_2 = n_updates_s[select_index_2]
select_mean_performance_1 = mean_performance_progress[select_index_1]
select_mean_performance_2 = mean_performance_progress[select_index_2]
print("number of training updates to reach {:.0f}% of saturation:".format(select_threshold_1 * 100),
      select_n_updates_1)
print("number of training updates to reach {:.0f}% of saturation:".format(select_threshold_2 * 100),
      select_n_updates_2)

# Plot
plots.configure_plot_style(style)

figsize_w = 5.4
figsize_h = 3.6
curve_color = plots.get_bar_color_with_group_id(group_id)
mean_lw = 1.0
trace_lw = 0.5
trace_alpha = 0.5
add_annotations = True
yaxis_formatter = plots.get_formatter_percent_of_optimal()
if width_argin is not None:
    if width_argin < figsize_w:
        yaxis_formatter = plots.get_formatter_percent()
        add_annotations = False
    figsize_w = width_argin
if height_argin is not None:
    figsize_h = height_argin

figsize = (figsize_w, figsize_h)

fig = plt.figure(figsize=figsize, constrained_layout=True)
ax = fig.gca()
ax.plot(n_updates_s,
     mean_performance_progress,
     '-',
     color=curve_color,
     lw=mean_lw)
ax.set_xlabel("Num. of weight updates")
ax.set_ylabel(plots.AXISLABEL_PERFORMANCE_SHORT)
ax.set_xlim(n_updates_s.min(), n_updates_s.max())
for i_network in range(n_networks):
    ax.plot(n_updates_s,
             performance_progress_s[i_network, :],
             '-',
             color=curve_color,
             alpha=trace_alpha,
             lw=trace_lw)
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(yaxis_formatter)
if add_annotations:
    select_label_1 = "≥{:.0f}%".format(select_threshold_1 * 100)
    select_label_2 = "≥{:.0f}%".format(select_threshold_2 * 100)
    arrow_length = (max_performance - min_performance) * 0.25
    ax.annotate(select_label_1,
                 (select_n_updates_1, select_mean_performance_1),
                 (select_n_updates_1, select_mean_performance_1 - arrow_length - 0.003),
                 horizontalalignment="center",
                 verticalalignment="top",
                 arrowprops = dict(arrowstyle="->", shrinkA=0., shrinkB=0.),
                 color="black",
                 fontsize = mpl.rcParams['legend.fontsize'])
    ax.annotate(select_label_2,
                 (select_n_updates_2, select_mean_performance_2),
                 (select_n_updates_2, select_mean_performance_2 - arrow_length - 0.003),
                 horizontalalignment="center",
                 verticalalignment="top",
                 arrowprops = dict(arrowstyle="->", shrinkA=0., shrinkB=0.),
                 color="black",
                 fontsize = mpl.rcParams['legend.fontsize'])
fig.savefig(out_path)


