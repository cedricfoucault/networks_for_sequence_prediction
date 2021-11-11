import argparse
import decoding_data
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.collections import LineCollection
import measure
import numpy as np
import pandas
import plots
from scipy import stats
import utils

predictor_key_dict = {
    "all": "hidden_all",
    "gates": "hidden_gates_tplus1",
    "state": "hidden_state",
}
confidence_label = "precision $R^2$ (%)"
prediction_label_dict = {
    "performance": "Performance\n(% of optimal log likelihood)",
    "logodds": "logodds-prediction $R^2$ (%)"
}

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="""data path with individual
                    prediction accuracy and confidence accuracy""")
parser.add_argument("figure_type", type=str, help="""type of figure to plot,
    either 'curve', 'scatter', or 'trajectory'""")
parser.add_argument("-o", "--output", help="path to output file")
args = parser.parse_args()
data_path = args.data_path
out_path = args.output
figure_type = args.figure_type
style = "presentation"


scale = "linear" if figure_type == "curve" or figure_type == "performance" else "log"
confidence_type = "all"
prediction_type = "performance" if figure_type == "performance" else "logodds"

df = pandas.read_csv(data_path)

network_ids = df["network_id"].unique()
n_training_updates_s = df["n_training_updates"].unique()
n_networks = len(network_ids)
n_n_training_updates = len(n_training_updates_s)

predictor_key = predictor_key_dict[confidence_type]
prediction_performances = np.empty((n_networks, n_n_training_updates))
prediction_logodds_r2s = np.empty((n_networks, n_n_training_updates))
confidence_r2s = np.empty((n_networks, n_n_training_updates))
df_predictor = df.query("predictor_key == @predictor_key")
for i_network, network_id in enumerate(network_ids):
    df_network = df_predictor.query("network_id == @network_id")
    for i_update, n_training_updates in enumerate(n_training_updates_s):
        df_for_entry = df_network.query(
            "n_training_updates == @n_training_updates"
        )
        prediction_performance = df_for_entry["prediction_performance"].to_numpy()
        prediction_logodds_r2 = df_for_entry["logodds_r2"].to_numpy()
        confidence_r2 = df_for_entry["confidence_r2"].to_numpy()
        assert len(prediction_performance) == 1
        assert len(prediction_logodds_r2) == 1
        assert len(confidence_r2) == 1
        prediction_performances[i_network, i_update] = prediction_performance
        prediction_logodds_r2s[i_network, i_update] = prediction_logodds_r2
        confidence_r2s[i_network, i_update] = confidence_r2

if prediction_type == "performance":
    prediction_values = prediction_performances
elif prediction_type == "logodds":
    prediction_values = prediction_logodds_r2s * 100.
confidence_values = confidence_r2s * 100.

prediction_min = prediction_values.min()
prediction_max = prediction_values.max()
confidence_min = confidence_values.min()
confidence_max = confidence_values.max()
prediction_stats = measure.get_summary_stats(prediction_values, axis=0)
confidence_stats = measure.get_summary_stats(confidence_values, axis=0)

n_train_updates_label = "training updates"
prediction_label = prediction_label_dict[prediction_type]

plots.configure_plot_style(style)

labelfontsize = 10
cmap = matplotlib.cm.get_cmap("plasma")
cbarorientation = "vertical"
cbaraspect = 100
cbarlabelpad = 12
if scale == "log":
        norm = matplotlib.colors.LogNorm(max(n_training_updates_s.min(), 1),
            n_training_updates_s.max())
else:
    norm = plt.Normalize(n_training_updates_s.min(),
        n_training_updates_s.max())

def plot_curve(ax, values, stats, xlabel, ylabel):
        ax.plot(n_training_updates_s,
             stats["mean"],
             '-',
             color=curve_color,
             lw=mean_lw)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(scale)
        if scale == "linear":
            ax.set_xlim(0, n_n_training_updates - 1)
        # elif scale == "log":
        #     ax.set_xlim(1, n_n_training_updates-1)
        for i_network in range(n_networks):
            ax.plot(n_training_updates_s,
                     values[i_network, :],
                     '-',
                     color=curve_color,
                     alpha=trace_alpha,
                     lw=trace_lw)
    

# Learning curve figure
if figure_type == "curve":
    figsize = (7.2, 3.6)
    curve_color = (75/255, 122/255., 176/255.)
    mean_lw = 3.
    trace_lw = 0.5
    trace_alpha = 0.5

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    axes = fig.subplots(nrows=1, ncols=2)
    # Prediction
    plot_curve(axes[0], prediction_values, prediction_stats, n_train_updates_label,  prediction_label)
    # Confidence
    plot_curve(axes[1], confidence_values, confidence_stats, n_train_updates_label, confidence_label)

    fig.savefig(out_path)

# Performance learning curve figure
if figure_type == "performance":
    figsize = (3.06, 2.87)
    curve_color = (75/255, 122/255., 176/255.)
    mean_lw = 3.
    trace_lw = 0.5
    trace_alpha = 0.5

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    plot_curve(fig.gca(), prediction_values, prediction_stats, n_train_updates_label,  prediction_label)
    fig.savefig(out_path)

# Trajectory figure
elif figure_type == "trajectory":
    figsize = (3.7, 3.56)
    mean_lw = 3.
    trace_lw = 0.5
    trace_alpha = 0.5

    def get_segments(x_values, y_values):
        points = np.array([x_values, y_values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments
    
    mean_segments = get_segments(prediction_stats["mean"], confidence_stats["mean"])
    linecollection = LineCollection(mean_segments,
                                    cmap=cmap,
                                    norm=norm)
    linecollection.set_array(n_training_updates_s)
    linecollection.set_linewidth(mean_lw)
    
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = plt.gca()
    line = ax.add_collection(linecollection)
    colorbar = fig.colorbar(line, ax=ax, orientation=cbarorientation, aspect=cbaraspect)
    colorbar.set_label(n_train_updates_label, rotation=270,
        labelpad=cbarlabelpad, fontsize=labelfontsize)

    plt.xlim((prediction_stats["mean"].min() + prediction_min) / 2,
             (prediction_stats["mean"].max() + prediction_max) / 2)
    plt.ylim((confidence_stats["mean"].min() + confidence_min) / 2,
             (confidence_stats["mean"].max() + confidence_max) / 2)
    plt.xlabel(prediction_label)
    plt.ylabel(confidence_label)
    for i_network in range(n_networks):
        segments = get_segments(
            prediction_values[i_network, :], confidence_values[i_network, :])
        linecollection = LineCollection(segments,
                                    cmap=cmap,
                                    norm=norm)
        linecollection.set_array(n_training_updates_s)
        linecollection.set_linewidth(trace_lw)
        linecollection.set_alpha(trace_alpha)
        ax.add_collection(linecollection)

    fig.savefig(out_path)


# Scatter plot figure
elif figure_type == "scatter":
    figsize = (3.7, 3.56)
    n_time_samples = 8 if scale == "log" else 16
    dotsize_individual = 2 if scale == "log" else 1.5
    dotsize_mean = 4.

    # correlation plot with individual networks and indexed by times
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = plt.gca()

    if scale == "log":
        time_base = n_n_training_updates ** (1 / (n_time_samples - 1))
        training_times = [int(np.floor(time_base ** i)) for i in range(n_time_samples)]
    else:
        training_times = np.arange(0, n_n_training_updates-1, step=(n_n_training_updates-1) // n_time_samples)
    for i_t, t in enumerate(training_times):
        cval = norm(t)
        color = cmap(cval)
        # individuals
        plt.plot(prediction_values[:, t],
                 confidence_values[:, t],
                 'o',
                 color=color,
                 markersize=dotsize_individual,
                 label=f"{t:d}")
        plt.xlabel(prediction_label)
        plt.ylabel(confidence_label)
        # mean
        show_mean = False
        if show_mean:
            plt.plot(prediction_stats["mean"][t],
                     confidence_stats["mean"][t],
                      'o',
                      color=color,
                      markersize=dotsize_mean,
                      mec="black")
    # time legend
    handles, labels = ax.get_legend_handles_labels()
    loc = "center left" if prediction_type == "performance" else "lower right"
    legend = ax.legend(reversed(handles), reversed(labels),
                       loc=loc, frameon=False,
               title=n_train_updates_label,
               handletextpad=0,
               fontsize=8)
    legend._legend_box.align = "left"
    legend.get_title().set_fontsize(8)
    
    r_scatter, p_value = stats.pearsonr(prediction_values[:, training_times].reshape(-1),
                                  confidence_values[:, training_times].reshape(-1))
    stat_label = utils.stat_label(p_value)
    plt.text(0.05, 1.0, f"r = {r_scatter:.3f}, {stat_label}",
             transform=ax.transAxes, fontsize=labelfontsize,
             va="top", ha="left")
    
    fig.savefig(out_path)
