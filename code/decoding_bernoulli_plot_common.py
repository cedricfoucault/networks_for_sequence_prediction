import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plots


def plot_dots_per_group(ax, values_per_group, median_per_group, labels):
    n_groups = len(values_per_group)
    y_groups = np.arange(n_groups)
    markersize_median = 5.
    markersize_sample = 1.5
    for i in range(n_groups):
        values = values_per_group[i]
        ax.scatter(values, np.ones(len(values)) * y_groups[i],
        marker='.', s=markersize_sample, color=plots.COLOR_DOT_SAMPLE)
        ax.scatter(median_per_group, y_groups,
            marker='o', s=markersize_median, color="#000000")
    ax.set_yticks(y_groups)
    ax.set_yticklabels(labels)

def format_func_correlation_optimal(value, tick_number, optimal_value):
    format_fun = plots.get_format_func_strip_leading_0_with_n_decimals(2)
    val_str = format_fun(value, tick_number)
    if value == optimal_value:
        return  val_str + "\n(optimal)"
    else:
        return val_str

def get_formatter_optimal_value(optimal_value):
    format_fun = lambda value, tick_number: format_func_correlation_optimal(value, tick_number, optimal_value)
    return plt.FuncFormatter(format_fun)