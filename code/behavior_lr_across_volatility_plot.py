import argparse
import matplotlib.pyplot as plt
import numpy as np
import plots
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="data file path")
parser.add_argument("-o_networks", "--output_networks", help="path to output file for networks")
parser.add_argument("-o_bayes", "--output_bayes", help="path to output file for Bayes")
args = parser.parse_args()
data_path = args.data_path
out_path_networks = args.output_networks
out_path_bayes = args.output_bayes


data_dict = torch.load(data_path)
volatility_denominator = data_dict["volatility_denominator"]
volatility_train_numerators = data_dict["volatility_train_numerators"]
volatility_test_numerators = data_dict["volatility_test_numerators"]
lr_mean_matrix = data_dict["lr_mean_matrix"]
lr_io_matrix = data_dict["lr_io_matrix"]

vmin = min(lr_mean_matrix.min(), lr_io_matrix.min())
vmax = max(lr_mean_matrix.max(), lr_io_matrix.max())

plots.configure_plot_style("paper")

double_width = 4.92
volatility_label = "change point probability"
figsize = (double_width / 2, double_width * 3 / 8)
labelfontsize = plt.rcParams["legend.fontsize"]
tickfontsize = labelfontsize
def plot_matrix(matrix, ylabel=f"Training\n{volatility_label}"):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.gca()
    imax = ax.imshow(matrix, aspect="equal",
                vmin=vmin, vmax=vmax,
                cmap=plt.cm.Greys,
    )
    ax.invert_yaxis()
    cbar = plt.colorbar(imax)
    cbar.ax.tick_params(labelsize=tickfontsize)
    ax.set_xlabel(f"Testing\n{volatility_label}", fontsize=labelfontsize)
    ax.set_ylabel(ylabel, fontsize=labelfontsize)
    x_tick_labels = [r"$\frac{{{:d}}}{{{:d}}}$".format(num, volatility_denominator) for num in volatility_test_numerators]
    y_tick_labels = [r"$\frac{{{:d}}}{{{:d}}}$".format(num, volatility_denominator) for num in volatility_train_numerators]
    ax.set_xticks(np.arange(len(x_tick_labels)))
    ax.set_xticklabels(x_tick_labels, fontsize=tickfontsize)
    ax.set_yticks(np.arange(len(y_tick_labels)))
    ax.set_yticklabels(y_tick_labels, fontsize=tickfontsize)
    ax.tick_params(axis='both', which='major', bottom=False, left=False)
    return fig


# Plot Network
fig = plot_matrix(lr_mean_matrix)
fig.savefig(out_path_networks)
print("Figure saved at", out_path_networks)

# Plot Bayes
fig = plot_matrix(lr_io_matrix, ylabel=f"Prior on\n{volatility_label}")
fig.savefig(out_path_bayes)#, bbox_inches="tight")
print("Figure saved at", out_path_bayes)

