import argparse
import decoding_data
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
import plots

parser = argparse.ArgumentParser()
parser.add_argument("decoder_group_dir", type=str, help="path to decoder group directory")
parser.add_argument('--include-conditional', dest='include_conditional', action='store_true')
parser.set_defaults(include_conditional=False)
parser.add_argument('--include-output', dest='include_output', action='store_true')
parser.set_defaults(include_output=False)
args = parser.parse_args()
decoder_group_dir = args.decoder_group_dir
include_conditional = args.include_conditional
include_output = args.include_output

out_dir = decoder_group_dir
regression_stats_path = decoding_data.get_group_regression_stats_path(decoder_group_dir)
regression_data_path = decoding_data.get_group_regression_data_path(decoder_group_dir)
df = pandas.read_csv(regression_stats_path)
df_individual = pandas.read_csv(regression_data_path)
network_ids = df_individual["network_id"].unique()
n_networks = len(network_ids)

key_label_dict = {
    "hidden_state": "state (t)",
    "hidden_gates_tplus1": "gates (t+1)",
    "hidden_all": "state (t)\nand\ngates (t+1)",
    "output": "output (t)",
    "io_confidence": "optimal confidence (t)",
    "io_evidence": "optimal log-odds (t)",
    "io_lr_tplus1": "optimal learning rate (t+1)",
    "io_confidence_0": r"$(1 | 0)$",
    "io_confidence_1": r"$(1 | 1)$",
    "io_confidence_xt": r"$(1 | x_{t})$",
    "io_confidence_1minusxt": r"$(1 | \overline{x_{t}})$",
    "io_evidence_0": r"$(1 | 0)$",
    "io_evidence_1": r"$(1 | 1)$",
    "io_evidence_xt": r"$(1 | x_{t})$",
    "io_evidence_1minusxt": r"$(1 | \overline{x_{t}})$",
    "io_lr_tplus1_0": r"$(1 | 0)$",
    "io_lr_tplus1_1": r"$(1 | 1)$",
    "io_lr_tplus1_xt": r"$(1 | x_{t})$",
    "io_lr_tplus1_1minusxt": r"$(1 | \overline{x_{t}})$",
}

plots.configure_plot_style()
color = "#6bbceb"

predictor_keys = decoding_data.get_confidence_predictor_keys(decoder_group_dir)
if include_output:
    predictor_keys.insert(0, "output")
n_groups = len(predictor_keys)
is_markov = "io_confidence_1" in df["outcome"].unique()
if is_markov:
# Markov case
    outcome_formats = ["{:}_0", "{:}_1"]
    if include_conditional:
        outcome_formats += ["{:}_xt", "{:}_1minusxt"]
    outcome_kinds = ["io_confidence", "io_evidence", "io_lr_tplus1"]
    outcome_keys_by_kind = [ [ s.format(kind) for s in outcome_formats] for kind in outcome_kinds ]
    for i_kind, kind in enumerate(outcome_kinds):
        
        outcome_keys = outcome_keys_by_kind[i_kind]
        n_outcomes = len(outcome_keys)
        figsize = (6.4, 4.8 * n_groups * n_outcomes / 3.)
        ypos = np.arange(n_outcomes)
        
        # fig = plt.figure(figsize=figsize)
        fig, axes = plt.subplots(nrows=n_groups, sharex=True,
                                 # figsize=figsize
                                 )
        plt.subplots_adjust(hspace=0.05)
        plt.xlabel(r"median variance explained ($r^2$)")
        plt.xlim(0, 100.2)
        plt.xticks(range(0, 101, 10))
        axes[-1].xaxis.set_major_formatter(plots.get_formatter_percent())
        
        plt.suptitle(key_label_dict[kind].capitalize())
        
        for i_predictor, predictor in enumerate(predictor_keys):
            r2_medians = np.empty((n_outcomes))
            r2_errors = np.empty((2, n_outcomes))
            r2_values = np.empty((n_outcomes, n_networks))
            labels = [key_label_dict[k] for k in outcome_keys]
            for i_outcome, outcome in enumerate(outcome_keys):
                row = df.query("outcome == @outcome & predictor == @predictor").iloc[0]
                r2_medians[i_outcome] = row["r2_median"]
                r2_errors[0, i_outcome] = r2_medians[i_outcome] - row["r2_median_ci_lower"]
                r2_errors[1, i_outcome] = row["r2_median_ci_upper"] - r2_medians[i_outcome]
                rows_individual = df_individual.query("outcome == @outcome & predictor == @predictor")
                r2_values[i_outcome, :] = rows_individual["r2"].to_numpy()
            r2_medians = r2_medians * 100.
            r2_errors = r2_errors * 100.
            r2_values = r2_values * 100.
            
            ax = axes[i_predictor] # plt.subplot(n_groups, 1, i_predictor + 1)
            # ax.barh(ypos, r2_medians, xerr=r2_errors, height=0.75, color=color, ecolor=ecolor)
            ax.barh(ypos, r2_medians, height=0.75, color=color)
            ax.plot(r2_values, ypos, '.', color=plots.COLOR_DOT_SAMPLE, markersize=1)
            ax.set_yticks(ypos)
            ax.set_yticklabels(labels)
            ax.set_xticks(range(0, 101, 10))
            ax.xaxis.set_major_formatter(plots.get_formatter_percent())
            plots.configure_bar_plot_axes(ax, labelbottom=(i_predictor == n_groups - 1))
            # Invert yaxis: Read top-to-bottom
            ax.invert_yaxis()
            ax.annotate(key_label_dict[predictor],
                        xy=(-0.25, 0.5), xytext=(-0.3, 0.5), xycoords='axes fraction', 
                        fontsize=12., ha='left', va='center',
                        # arrowprops=dict(arrowstyle='-[, widthB=10.0, lengthB=1.5', lw=2.0)
                        )
        out_filename_prefix = f"decoding_net_to_{kind}"
        if include_conditional:
            out_filename_prefix += "_with-conditional"
        if include_output:
            out_filename_prefix += "_with-output"
        out_path = os.path.join(out_dir, out_filename_prefix + ".png")
        fig.savefig(out_path, bbox_inches='tight')
        print("Figure saved at", out_path)
# Bernoulli case
else:
    ypos = np.arange(n_groups)
    figsize = (6.4, 4.8 * n_groups / 3.)
    outcome_keys = ["io_confidence", "io_evidence", "io_lr_tplus1"]
    
    for outcome_key in outcome_keys:
        
        r2_medians = np.empty((n_groups))
        r2_errors = np.empty((2, n_groups))
        r2_values = np.empty((n_groups, n_networks))
        labels = [key_label_dict[k] for k in predictor_keys]
        for i, predictor in enumerate(predictor_keys):
            row = df.query("outcome == @outcome_key & predictor == @predictor").iloc[0]
            r2_medians[i] = row["r2_median"]
            r2_errors[0, i] = r2_medians[i] - row["r2_median_ci_lower"]
            r2_errors[1, i] = row["r2_median_ci_upper"] - r2_medians[i]
            rows_individual = df_individual.query("outcome == @outcome_key & predictor == @predictor")
            r2_values[i, :] = rows_individual["r2"].to_numpy()
        r2_medians = r2_medians * 100. # percent value
        r2_errors = r2_errors * 100. # percent value
        r2_values = r2_values * 100. # percent value
       
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        # plt.barh(ypos, r2_medians, xerr=r2_errors, height=0.75, color=color, ecolor=ecolor)
        plt.barh(ypos, r2_medians, height=0.75, color=color)
        plt.plot(r2_values, ypos, '.', color=plots.COLOR_DOT_SAMPLE, markersize=1)
        plt.yticks(ypos, labels=labels)
        plt.xlabel(r"median variance explained ($r^2$)")
        plt.xlim(0, 100.2)
        ax.set_xticks(range(0, 101, 10))
        ax.xaxis.set_major_formatter(plots.get_formatter_percent())
        plots.configure_bar_plot_axes(ax)
        # Invert yaxis: Read top-to-bottom
        ax.invert_yaxis()
        
        plt.title(key_label_dict[outcome_key].capitalize())
        
        out_filename_prefix = f"decoding_net_to_{outcome_key}"
        if include_output:
            out_filename_prefix += "_with-output"
        out_filename = f"{out_filename_prefix}.png"
        out_path = os.path.join(out_dir, out_filename)
        fig.savefig(out_path, bbox_inches='tight')
        print("Figure saved at", out_path)
