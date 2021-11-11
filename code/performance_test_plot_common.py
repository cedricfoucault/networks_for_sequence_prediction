import numpy as np
import measure
import plots
import training_data

def get_performances_labels_colors(df, group_ids, legend_style="default", detailed_label=True, test_gen_process=None):
    n_groups = len(group_ids)
    chance_loss = df["chance_loss"].iloc[0]
    io_loss = df["io_loss"].iloc[0]
    n_models = training_data.get_n_models(df["group_id"].iloc[0])
    labels = []
    colors = []
    sublabels = []
    performances = np.empty((n_groups, n_models))
    for i, group_id in enumerate(group_ids):
        # get label
        models = training_data.load_models(group_id)
        label = plots.get_label_with_group_id(group_id, legend_style, detailed=detailed_label)
        labels.append(label)
        sublabel = plots.get_sublabel_with_group_id(group_id, legend_style)
        sublabels.append(sublabel)
        # get color
        color = plots.get_bar_color_with_group_id(group_id, legend_style, test_gen_process=test_gen_process, metric_kind="performance")
        colors.append(color)
        # get individual data point values
        losses = df.query("group_id == @group_id")["loss"].to_numpy()
        performances[i, :] = measure.get_percent_value(losses, chance_loss, io_loss)
    return performances, labels, sublabels, colors

def plot_performances_barh(ax, performances, labels, sublabels, colors,
    bar_fontsize, xlabel, xticks=range(0, 101, 10), optimal_label="optimal",
    ypos=None, bar_height=plots.BAR_HEIGHT):
    performance_means = performances.mean(axis=1)
    if ypos is None:
        n_groups = performances.shape[0]
        ypos = np.arange(n_groups)
        ax.invert_yaxis() # Invert yaxis: Read top-to-bottom
    ax.barh(ypos, performance_means, height=bar_height, color=colors)
    ax.plot(performances, ypos, '.', color=plots.COLOR_DOT_SAMPLE, markersize=1.5)
    for i, sublabel in enumerate(sublabels):
        sublabel = sublabels[i]
        if sublabel:
            ax.text(1, ypos[i], sublabel, va="center", ha="left")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=bar_fontsize)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_xlim(0, 100.1)
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(plots.get_formatter_percent_of_optimal(optimal_label))
    plots.configure_bar_plot_axes(ax)