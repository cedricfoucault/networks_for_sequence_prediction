import argparse
import decoding_data
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
import plots

parser = argparse.ArgumentParser()
parser.add_argument("decoder_group_dir", type=str, help="path to decoder group directory")
parser.add_argument("-o_logodds", "--output_logodds", help="path to output file for log odds")
parser.add_argument("-o_confidence", "--output_confidence", help="path to output file for confidence")
parser.add_argument("--style", default="paper", help="format style (paper, presentation)")
args = parser.parse_args()
decoder_group_dir = args.decoder_group_dir
out_path_logodds = args.output_logodds
out_path_confidence = args.output_confidence
style = args.style

regression_stats_path = decoding_data.get_group_regression_stats_path(decoder_group_dir)
regression_data_path = decoding_data.get_group_regression_data_path(decoder_group_dir)
df = pandas.read_csv(regression_stats_path)
df_individual = pandas.read_csv(regression_data_path)
network_ids = df_individual["network_id"].unique()
n_networks = len(network_ids)

key_label_dict = decoding_data.KEY_LABEL_DICT
task_kind = "markov"
outcome_kinds = ["confidence", "logodds"]
predictor_keys_by_kind = [ decoding_data.get_predictor_keys(decoder_group_dir, outcome_kind) for outcome_kind in outcome_kinds ]
outcome_keys_by_kind = [ decoding_data.get_outcome_keys(decoder_group_dir, task_kind, outcome_kind) for outcome_kind in outcome_kinds ]
out_path_by_kind = [out_path_confidence, out_path_logodds]

plots.configure_plot_style(style)
confidence_color = plots.COLOR_BAR_CONFIDENCE
logodds_color = plots.COLOR_BAR_LOGODDS
labelfontsize = 10
subplot_adjust_kwargs_by_kind = [
    dict(left=0.27, right=0.95, bottom=0.29, top=0.99),
    dict(left=0.27, right=0.95, bottom=0.57, top=0.99)
    ]
figsize_by_kind = [(6.56, 2.05), (6.56, 1.04)]
xlabel = plots.AXISLABEL_R2
color_by_kind = [confidence_color, logodds_color]
x_text = -0.1
if style.lower() == "presentation":
    figsize_by_kind = [(6.8, 2.4), (6.8, 1.34)]
    labelfontsize = 14.
    subplot_adjust_kwargs_by_kind = [
    dict(left=0.345, right=0.95, bottom=0.327, top=0.99),
    dict(left=0.345, right=0.95, bottom=0.58, top=0.99)
    ]
    x_text = -0.12


for i_kind, kind in enumerate(outcome_kinds):
# for i_kind, kind in enumerate(outcome_kinds[0:1]):
    figsize = figsize_by_kind[i_kind]
    outcome_keys = outcome_keys_by_kind[i_kind]
    predictor_keys = predictor_keys_by_kind[i_kind]
    out_path = out_path_by_kind[i_kind]
    n_predictors = len(predictor_keys)
    n_outcomes = len(outcome_keys)
    n_bars = n_predictors * n_outcomes
    
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    plt.subplots_adjust(**subplot_adjust_kwargs_by_kind[i_kind])
    
    ypos = np.arange(n_bars)
    # ax = axes[i_kind]
    labels = [key_label_dict[k] for _ in predictor_keys for k in outcome_keys]
    colors = [ color_by_kind[i_kind] for i in range(n_bars) ]
    r2_medians = np.empty((n_bars))
    r2_values = np.empty((n_bars, n_networks))
    for i, predictor in enumerate(predictor_keys):
        for j, outcome in enumerate(outcome_keys):
            row = df.query("outcome == @outcome & predictor == @predictor").iloc[0]
            r2_medians[i*n_outcomes + j] = row["r2_median"]
            rows_individual = df_individual.query("outcome == @outcome & predictor == @predictor")
            r2_values[i*n_outcomes + j, :] = rows_individual["r2"].to_numpy()
    r2_medians = r2_medians * 100. # percent value
    r2_values = r2_values * 100. # percent value

    ax.barh(ypos, r2_medians, height=plots.BAR_HEIGHT, color=colors)
    ax.plot(r2_values, ypos, '.', color=plots.COLOR_DOT_SAMPLE, markersize=1.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    plt.xlim(0, 100.2)
    ax.set_xticks(range(0, 101, 25))
    ax.set_xticks(np.arange(0, 101, 12.5), minor=True)
    ax.xaxis.set_major_formatter(plots.get_formatter_percent())
    plots.configure_bar_plot_axes(ax, show_minor=True)
    # Emphasize the "large" tick
    ax.get_xticklabels()[1].get_fontproperties().set_weight('bold')
    ax.get_xgridlines()[1].set_alpha(1)
    # Annotate with predictor
    for i, predictor in enumerate(predictor_keys):
        y_text = 1 - (i * n_outcomes + 1) / (n_outcomes * n_predictors)
        ax.annotate(key_label_dict[predictor],
                            xy=(x_text + 0.02, y_text), xytext=(x_text, y_text), xycoords='axes fraction', 
                            fontsize=labelfontsize, ha='right', va='center',
                            )
    # Invert yaxis: Read top-to-bottom
    ax.invert_yaxis()
    
    fig.savefig(out_path)
    print("Figure saved at", out_path)
