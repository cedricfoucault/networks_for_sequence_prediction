import argparse
import decoding_data
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
import plots

parser = argparse.ArgumentParser()
parser.add_argument("decoder_group_dir", type=str, help="path to decoder group directory")
parser.add_argument("-o", "--output", help="path to output file")
parser.add_argument("--style", default="paper", help="format style (paper, presentation)")
args = parser.parse_args()
decoder_group_dir = args.decoder_group_dir
out_path = args.output
style = args.style

regression_stats_path = decoding_data.get_group_regression_stats_path(decoder_group_dir)
regression_data_path = decoding_data.get_group_regression_data_path(decoder_group_dir)
df = pandas.read_csv(regression_stats_path)
df_individual = pandas.read_csv(regression_data_path)
network_ids = df_individual["network_id"].unique()
n_networks = len(network_ids)

key_label_dict = decoding_data.KEY_LABEL_DICT

r2_bayes = 0.04

plots.configure_plot_style(style)
bayes_color = plots.COLOR_BAR_ALTERNATIVE
network_color = plots.COLOR_BAR_CONFIDENCE
figsize = (5.12, 2.65)
adjust_left = 0.28
adjust_bottom = 0.23
adjust_top = 0.99
adjust_right = 0.96
xlabel = plots.AXISLABEL_R2
if style.lower() == "presentation":
    figsize = (6.6, 2.65)
    adjust_bottom = 0.30
    adjust_left = 0.30

predictor_keys = decoding_data.get_confidence_predictor_keys(decoder_group_dir)
outcome_key = "io_confidence"
n_groups = len(predictor_keys)


ypos = np.arange(n_groups + 1)
labels = ["Bayes distribution"] + [key_label_dict[k] for k in predictor_keys]
colors = [bayes_color] + [network_color for _ in range(n_groups)]
r2_medians = np.empty((n_groups + 1))
r2_values = np.empty((n_groups + 1, n_networks))
r2_medians[0] = r2_bayes
r2_values[0, :] = r2_bayes
for i, predictor in enumerate(predictor_keys):
    row = df.query("outcome == @outcome_key & predictor == @predictor").iloc[0]
    r2_medians[i+1] = row["r2_median"]
    rows_individual = df_individual.query("outcome == @outcome_key & predictor == @predictor")
    r2_values[i+1, :] = rows_individual["r2"].to_numpy()
r2_medians = r2_medians * 100. # percent value
r2_values = r2_values * 100. # percent value
    
fig = plt.figure(figsize=figsize)
plt.subplots_adjust(left=adjust_left, bottom=adjust_bottom, top=adjust_top, right=adjust_right)
ax = fig.gca()
plt.barh(ypos, r2_medians, height=plots.BAR_HEIGHT, color=colors)
plt.plot(r2_values[1:], ypos[1:], '.', color=plots.COLOR_DOT_SAMPLE, markersize=1.5)
plt.yticks(ypos, labels=labels)
plt.xlabel(xlabel)
plt.xlim(0, 100.2)
ax.set_xticks(range(0, 101, 25))
ax.set_xticks(np.arange(0, 101, 12.5), minor=True)
ax.xaxis.set_major_formatter(plots.get_formatter_percent())
# Provide tick lines along the xticks
plots.configure_bar_plot_axes(ax, show_minor=True)
# Invert yaxis: Read top-to-bottom
ax.invert_yaxis()
# Emphasize the "large" tick
ax.get_xticklabels()[1].get_fontproperties().set_weight('bold')
ax.get_xgridlines()[1].set_alpha(1)
        
fig.savefig(out_path)
print("Figure saved at", out_path)
