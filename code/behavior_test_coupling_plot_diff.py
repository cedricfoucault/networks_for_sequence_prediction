import argparse
from behavior_test_coupling_common import get_changes_in_prediction_by_coupled
import matplotlib as mpl
import matplotlib.pyplot as plt
import measure
import numpy as np
import pandas
import plots
import scipy.stats as stats
import training_data
import utils

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="path to CSV data file")
parser.add_argument("data_path_ideal_observer", type=str,
    help="path to CSV data file for the ideal observer data")
utils.add_arguments(parser, ["output", "style", "width", "height"])
args = parser.parse_args()
out_path = args.output
style = args.style
width_argin = args.width
height_argin = args.height

df = pandas.read_csv(args.data_path)
df_io = pandas.read_csv(args.data_path_ideal_observer)
pre_rep_pgen = df["pre_rep_pgen"].iloc[0]
pre_other_pgen_values = df["pre_other_pgen"].unique()
assert(len(pre_other_pgen_values) == 2)
rep_lengths = df["rep_length"].unique()
n_rep_lengths = len(rep_lengths)

# Plot
plots.configure_plot_style(style)

legendfontsize = plt.rcParams["legend.fontsize"]
stat_fontsize = legendfontsize
n_xtick_values = 6
show_axes_labels = True
show_legend = False
color_by_coupled = { False: plots.COLOR_LINE_INDEPENDENT, True: plots.COLOR_LINE_COUPLED }

fig_w = 3.16
fig_h = 2.43
if width_argin is not None:
    fig_w = width_argin
if height_argin is not None:
    if height_argin < fig_h:
        show_axes_labels = False
        n_xtick_values = 3
    fig_h = height_argin
    
figsize = (fig_w, fig_h)

fig = plt.figure(figsize=figsize, constrained_layout=True)
ax = fig.gca()

network_p_diff_s_by_coupled = get_changes_in_prediction_by_coupled(df)
io_p_diff_s_by_coupled = get_changes_in_prediction_by_coupled(df_io)

# t-test between networks coupled and independent
p_vals = np.empty((n_rep_lengths))
for i_rep_length, rep_length in enumerate(rep_lengths):
    t_val, p_val = stats.ttest_ind(
        network_p_diff_s_by_coupled[True][i_rep_length, :],
        network_p_diff_s_by_coupled[False][i_rep_length, :]
        )
    p_val = p_val / 2 if t_val >= 0 else 1 # one tailed
    p_vals[i_rep_length] = p_val

# Plot
i_start_plot = 2
for is_coupled in [True, False]:
    io_p_diff_s = io_p_diff_s_by_coupled[is_coupled]
    assert io_p_diff_s.shape[1] == 1, "there should be only 1 ideal observer model"
    io_p_diff = io_p_diff_s[:, 0]
    network_p_diff_s = network_p_diff_s_by_coupled[is_coupled]
    network_p_diff_stats = measure.get_summary_stats(network_p_diff_s, axis=1)

    if show_legend:
        group_id_coupled = df["group_id_coupled"].iloc[0]
        modelkind_label = training_data.load_models(group_id_coupled)[0].get_label(detailed=False)
        coupled_label = "coupled" if is_coupled else "independent"
        models_label = f"{modelkind_label}s {coupled_label}"
        io_label = f"Optimal for {coupled_label}"
        label_pad = 1.5
    # networks
    ax.plot(rep_lengths[i_start_plot:], network_p_diff_stats["mean"][i_start_plot:], '-',
        color=color_by_coupled[is_coupled])
    ax.fill_between(rep_lengths[i_start_plot:],
        network_p_diff_stats["ci_lower"][i_start_plot:], network_p_diff_stats["ci_upper"][i_start_plot:],
        alpha=plots.ALPHA_CI, color=color_by_coupled[is_coupled])
    if show_legend:
        ax.text(rep_lengths[-1] + label_pad, network_p_diff_stats["mean"][-1],
            models_label,
            va="center", ha="left",
            color=color_by_coupled[is_coupled],
            fontsize=legendfontsize)
    # ideal observer
    ax.plot(rep_lengths[i_start_plot:], io_p_diff[i_start_plot:], ':',
            color=color_by_coupled[is_coupled])
    if show_legend:
        ax.text(rep_lengths[-1] + label_pad, io_p_diff[-1],
            io_label,
            color=color_by_coupled[is_coupled],
            va="center", ha="left",
            fontsize=legendfontsize)

xtick_values = np.linspace(rep_lengths[i_start_plot], rep_lengths[-1], num=n_xtick_values)
xtick_values = np.floor(xtick_values)
ax.set_xticks(xtick_values)
ax.set_xlim(xtick_values[0], xtick_values[-1])
if show_axes_labels:
    ax.set_xlabel("Streak length")
    ax.set_ylabel("$\mathregular{| p_{after} - p_{before}} |}$")
else:
    ax.set_xlabel("")
    ax.set_ylabel("")

ymin, ymax = ax.get_ylim()
stat_y = ymax
stat_color = "black"
crit = 0.001
i_test_pass_first = -1
i_test_pass_last = -1
for i_num_bigrams, num_bigrams in enumerate(rep_lengths):
    p_val = p_vals[i_num_bigrams]
    if i_num_bigrams == 0 or np.isnan(p_val):
        continue
    if (p_val <= crit):
        if i_test_pass_first == -1:
            i_test_pass_first = i_num_bigrams
    else:
        if i_test_pass_first != -1:
            i_test_pass_last = i_num_bigrams - 1
            break
if i_test_pass_last == -1 and i_test_pass_first != -1:
    i_test_pass_last = i_num_bigrams
if i_test_pass_first != -1:
    stat_label = utils.stat_label(crit)
    stat_x1 = rep_lengths[i_test_pass_first]
    stat_x2 = rep_lengths[i_test_pass_last]
    va = "baseline"
else:
    stat_label = utils.stat_label(1)
    stat_x1, stat_x2 = xtick_values[0], xtick_values[-1]
    va = "bottom"

ax.plot([stat_x1, stat_x2], [stat_y, stat_y],
                 linewidth=.5, color=stat_color)
ax.text((stat_x1 + stat_x2) / 2, stat_y, stat_label, ha='center', va=va,
         color=stat_color, fontsize=stat_fontsize)


fig.savefig(out_path)
print("Figure saved at", out_path)
