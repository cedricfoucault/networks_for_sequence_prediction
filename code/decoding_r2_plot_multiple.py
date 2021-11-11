import argparse
import data
import decoding_data
import numpy as np
import pandas
import plots
import matplotlib.pyplot as plt
import utils

# map each predictor key to a list
# where each element of the list is a dictionary
# that contains all the pieces needed to plot one group
# (median r2, individual r2s, label, color)
def get_bar_data_s_by_predictor(decoder_group_dirs, task_kind, outcome_kind, legend_style):
    bar_data_s_by_predictor = { }
    for decoder_group_dir in decoder_group_dirs:
        regression_stats_path = decoding_data.get_group_regression_stats_path(decoder_group_dir)
        regression_data_path = decoding_data.get_group_regression_data_path(decoder_group_dir)
        df_stats = pandas.read_csv(regression_stats_path)
        df_individual = pandas.read_csv(regression_data_path)
        network_ids = df_individual["network_id"].unique()
        n_networks = len(network_ids)
        test_gen_process = data.load_gen_process(df_individual["dataset_name"].iloc[0])

        group_id = df_stats["group_id"].iloc[0]
        label = plots.get_hidden_layer_label_with_group_id(group_id, legend_style=legend_style)
        color = plots.get_bar_color_with_group_id(group_id,
            legend_style=legend_style, test_gen_process=test_gen_process, metric_kind=outcome_kind)

        predictor_keys = decoding_data.get_predictor_keys(decoder_group_dir, outcome_kind)
        outcome_keys = decoding_data.get_outcome_keys(decoder_group_dir, task_kind, outcome_kind)
        for predictor_key in predictor_keys:
            for i_outcome, outcome_key in enumerate(outcome_keys):
                row = df_stats.query("outcome == @outcome_key & predictor == @predictor_key").iloc[0]
                rows_individual = df_individual.query("outcome == @outcome_key & predictor == @predictor_key")
                r2_median = row["r2_median"] * 100. # percent value
                r2_values = rows_individual["r2"].to_numpy() * 100 # percent value
                # bar_data = dict(label=label, color=color,
                #     r2_median=r2_median, r2_values=r2_values)
                bar_data = dict(color=color, r2_median=r2_median, r2_values=r2_values)
                if i_outcome == 0:
                    bar_data["label"] = label
                if len(outcome_keys) > 1:
                    bar_data["sublabel"] = decoding_data.KEY_LABEL_DICT[outcome_key]
                if predictor_key not in bar_data_s_by_predictor:
                    bar_data_s_by_predictor[predictor_key] = [bar_data]
                else:
                    bar_data_s_by_predictor[predictor_key].append(bar_data)

    return bar_data_s_by_predictor

def plot_decoding_r2_figure(bar_data_s_by_predictor, outcome_kind, style):
    ordered_predictor_keys = decoding_data.get_all_predictor_keys(outcome_kind)
    predictor_keys = utils.list_members_of_other_list(
        ordered_predictor_keys,
        bar_data_s_by_predictor.keys())
    n_predictors = len(predictor_keys)
    n_bars_total = 0 
    n_bars_by_predictor = {}
    for predictor_key, bar_data_s in bar_data_s_by_predictor.items():
        n_bars = len(bar_data_s)
        n_bars_by_predictor[predictor_key] = n_bars
        n_bars_total += n_bars
    height_per_bar = (0.99-0.3) * 2.65 / 4
    w_pad = height_per_bar
    height_margin = 0.5
    height = height_per_bar * n_bars_total + w_pad * (n_predictors) + height_margin
    wspace = 0
    xlabel = plots.AXISLABEL_R2
    height_ratios = [ n_bars_by_predictor[k] / n_bars_total for k in predictor_keys ]
    figsize = (5.12, height)
    if style.lower() == "presentation":
        figsize = (6.6, height)

    plots.configure_plot_style(style)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gridspec = fig.add_gridspec(nrows=n_predictors, ncols=1,
        height_ratios=height_ratios)
    ax_shared = None

    for i_predictor, predictor_key in enumerate(predictor_keys):
        bar_data_s = bar_data_s_by_predictor[predictor_key]
        n_bars = len(bar_data_s)
        ypos = np.arange(n_bars)
        colors = [bar_data["color"] for bar_data in bar_data_s]
        r2_medians = [bar_data["r2_median"] for bar_data in bar_data_s]

        if not ax_shared:
            ax_shared = fig.add_subplot(gridspec[i_predictor])
            ax = ax_shared
        else:
            ax = fig.add_subplot(gridspec[i_predictor], sharex=ax_shared)
        ax.barh(ypos, r2_medians, height=plots.BAR_HEIGHT, color=colors)
        for i_bar, bar_data in enumerate(bar_data_s):
            r2_values = bar_data["r2_values"]
            if "sublabel" in bar_data:
                ax.text(1, i_bar, bar_data["sublabel"], va="center", ha="left")
            ax.plot(r2_values, i_bar * np.ones(len(r2_values)), '.', color=plots.COLOR_DOT_SAMPLE, markersize=1.5)

        yticks = []
        yticklabels = []
        for i_bar, bar_data in enumerate(bar_data_s):
            if "label" in bar_data:
                yticks.append(i_bar)
                yticklabels.append(bar_data["label"])
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        plots.configure_bar_plot_axes(ax, show_minor=True)
        # Invert yaxis: Read top-to-bottom
        ax.invert_yaxis()
        ax.set_title(decoding_data.PREDICTOR_LABEL_DICT[predictor_key])

    ax.set_xlabel(xlabel)
    ax.set_xlim(0, 100.2)
    ax.set_xticks(range(0, 101, 25))
    ax.set_xticks(np.arange(0, 101, 12.5), minor=True)
    ax.xaxis.set_major_formatter(plots.get_formatter_percent())
    return fig

parser = argparse.ArgumentParser()
parser.add_argument("decoder_group_dirs", nargs="+", type=str, help="path to decoder group directories")
parser.add_argument("-o", "--output", help="path to output file")
parser.add_argument("-o_kind", "--outcome_kind", help="kind of outcome (confidence, logodds)")
parser.add_argument("-t_kind", "--task_kind", help="kind of task (bernoulli, markov)")
parser.add_argument("--style", default="paper", help="format style (paper, presentation)")
parser.add_argument("--legend_style", default="architecture")

args = parser.parse_args()
decoder_group_dirs = args.decoder_group_dirs
out_path = args.output
task_kind = args.task_kind
outcome_kind = args.outcome_kind
style = args.style
legend_style = args.legend_style

bar_data_s_by_predictor = get_bar_data_s_by_predictor(decoder_group_dirs, task_kind, outcome_kind, legend_style)
fig = plot_decoding_r2_figure(bar_data_s_by_predictor, outcome_kind, style)     
fig.savefig(out_path)
print("Figure saved at", out_path)
