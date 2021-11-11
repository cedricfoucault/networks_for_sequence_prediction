import argparse
import decoding_data
import generate
import io_model
import matplotlib.pyplot as plt
import measure
from measure import get_decoded_tensor
import os
import plots
import scipy.stats as stats
import sequence
import torch
import training_data
import utils

seed = 47
utils.set_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("group_id", type=str, help="group id of the decoders to plot")
parser.add_argument("decoder_group_dir", type=str, help="path to decoder group directory")
parser.add_argument("-o", "--output", help="path to output file")
parser.add_argument("--style", default="paper", help="format style (paper, presentation)")
args = parser.parse_args()
group_id = args.group_id
decoder_group_dir = args.decoder_group_dir
out_path = args.output
style = args.style

predictor_key = "hidden_state"
confidence_0_key, confidence_1_key = "io_confidence_0", "io_confidence_1"
evidence_0_key, evidence_1_key = "io_evidence_0", "io_evidence_1"

networks, network_ids = training_data.load_models_ids(group_id)
confidence_0_decoders = decoding_data.load_decoders(decoder_group_dir, confidence_0_key, predictor_key, network_ids)
confidence_1_decoders = decoding_data.load_decoders(decoder_group_dir, confidence_1_key, predictor_key, network_ids)
evidence_0_decoders = decoding_data.load_decoders(decoder_group_dir, evidence_0_key, predictor_key, network_ids)
evidence_1_decoders = decoding_data.load_decoders(decoder_group_dir, evidence_1_key, predictor_key, network_ids)

# Generate sequence of inputs to reveal differences in the two transition probabilities
try:
  p_change = training_data.load_gen_process(group_id).p_change
except KeyError:
  p_change = 1 / 75. # in case model wasn't trained

gen_process = generate.GenerativeProcessMarkovIndependent(p_change)
mean_stable_time = 1 / gen_process.p_change
reversal_times_0 = [round(mean_stable_time * 2/3)]
reversal_times_1 = [round(mean_stable_time * 4/3)]
n_time_steps = round(mean_stable_time * 2)
n_samples = 1
inputs = torch.empty((n_time_steps, n_samples, 1))
p_gen_0 = 0.8
p_gen_1 = 0.8
obs_tminus1 = 0
for t in range(n_time_steps):
    p = p_gen_1 if obs_tminus1 == 1 else p_gen_0
    inputs[t, 0, 0] = stats.bernoulli.rvs(p)
    obs_tminus1 = inputs[t, 0, 0]
    if t in reversal_times_0:
        p_gen_0 = 1 - p_gen_0
    if t in reversal_times_1:
        p_gen_1 = 1 - p_gen_1
        

# Get the network with median decoding performance for one of the logodds
median_network_id = decoding_data.get_median_r2_network_id(decoder_group_dir, evidence_0_key, predictor_key)
i_network = list(network_ids).index(median_network_id)

# Compute decoded transition probabilities and sds
hidden_states = networks[i_network].forward_activations(inputs)["hidden_state"]
confidence_0 = get_decoded_tensor(confidence_0_decoders[i_network], hidden_states)
confidence_1 = get_decoded_tensor(confidence_1_decoders[i_network], hidden_states)
evidence_0 = get_decoded_tensor(evidence_0_decoders[i_network], hidden_states)
evidence_1 = get_decoded_tensor(evidence_1_decoders[i_network], hidden_states)
sds_0 = measure.get_sds_from_confidence(confidence_0)
sds_1 = measure.get_sds_from_confidence(confidence_1)
ps_0 = measure.get_ps_from_logits(evidence_0)
ps_1 = measure.get_ps_from_logits(evidence_1)

# Compute Bayes transition probabilities and sds
io = io_model.IOModel(gen_process)
io_output_stats = io.get_output_stats(inputs)
io_sds_0 = io_output_stats["sd"][:, :, 0]
io_sds_1 = io_output_stats["sd"][:, :, 1]
io_ps_0 = io_output_stats["mean"][:, :, 0]
io_ps_1 = io_output_stats["mean"][:, :, 1]

# Plot

plots.configure_plot_style(style)
figsize = (6.56, 1.5)
linewidth = 1
time_steps = range(n_time_steps)
cmap = plt.cm.PuOr
COLOR_OBSERVATION = plots.COLOR_OBSERVATION
# color_0 = cmap(0.2)
# color_1 = cmap(0.9)
color_0 = "#FFCF00"
color_1 = "#3415B0"
alpha_sd = 0.5
labelfontsize = plt.rcParams["legend.fontsize"]
network_title = "Bigram readouts from the recurrent activity"
bayes_ylabel = ""
adjust_left = 0.071
adjust_right = 0.97
wspace = 0.1
sharey = True
if style.lower() == "presentation":
  figsize = (11.5, 2.04)
  bayes_ylabel = "p $\pm$ sd"
  adjust_left = 0.055
  adjust_right = 0.97
  wspace = 0.2

def plot_p_sd(ps_0, sds_0, ps_1, sds_1, ax, title="", ylabel=""):
    ax.plot(time_steps, sequence.get_numpy1D_from_tensor(inputs), 'o', color=COLOR_OBSERVATION, markersize=.5)
    ax.plot(sequence.get_numpy1D_from_tensor(1 - ps_0), '-', label="0|0", color=color_0, lw=linewidth)
    ax.fill_between(time_steps,
                      sequence.get_numpy1D_from_tensor(1 - ps_0 - sds_0),
                      sequence.get_numpy1D_from_tensor(1 - ps_0 + sds_0),
                      color=color_0, alpha=alpha_sd)
    ax.plot(measure.get_vector(ps_1),'-', label="1|1", color=color_1, lw=linewidth)
    ax.fill_between(time_steps,
                      sequence.get_numpy1D_from_tensor(ps_1 - sds_1),
                      sequence.get_numpy1D_from_tensor(ps_1 + sds_1),
                      color=color_1, alpha=alpha_sd)
    ax.text(n_time_steps + 1, 1 - ps_0[-1:, :], "0|0", fontsize=labelfontsize, color=color_0)
    ax.text(n_time_steps + 1, ps_1[-1:, :], "1|1", fontsize=labelfontsize, color=color_1)
    ax.set_xlim(-1, n_time_steps)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel(plots.AXISLABEL_TIME)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=sharey)#, constrained_layout=True)
plt.subplots_adjust(left=adjust_left, right=adjust_right, bottom=0.24, top=0.87, wspace=wspace, hspace=0)
plot_p_sd(ps_0, sds_0, ps_1, sds_1, axes[0], title=network_title, ylabel="p $\pm$ sd")
plot_p_sd(io_ps_0, io_sds_0, io_ps_1, io_sds_1, axes[1], title="Optimal estimates", ylabel=bayes_ylabel)
# plt.subplots_adjust(hspace=0.3)

# fig.savefig(os.path.join(out_dir, out_fname), bbox_inches="tight")
fig.savefig(out_path)



