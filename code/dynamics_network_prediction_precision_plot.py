import argparse
import decoding_data
from dynamics_common import (T_PLOT_START, COLOR_BY_LEVEL, DEFAULT_SEED,
    get_inputs_by_level_with_seed, print_inputs_by_level)
import matplotlib.pyplot as plt
import numpy as np
import plots
import training_data
import utils

def _get_hidden_state(network, inputs):
    hidden_state = network.forward_activations(inputs)["hidden_state"]
    hidden_state = hidden_state.squeeze().detach().numpy()
    return hidden_state

def get_state_component(state, w_proj):
    return (state @ w_proj.T).squeeze()

parser = argparse.ArgumentParser()
parser.add_argument("--group_id", type=str, help="trained model group id")
parser.add_argument("--decoder_group_dir", type=str, help="path to decoder group directory")
parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
utils.add_arguments(parser, ["output", "width", "height"])
parser.add_argument("--verbose", action="store_true", default=False)
args = parser.parse_args()
group_id = args.group_id
decoder_group_dir = args.decoder_group_dir
out_path = args.output

inputs_by_level = get_inputs_by_level_with_seed(args.seed)
if args.verbose:
    print_inputs_by_level(inputs_by_level)

figsize = args.width, args.height
plots.configure_plot_style("paper")

networks, network_ids = training_data.load_models_ids(group_id)
precision_decoders = decoding_data.load_decoders(decoder_group_dir, "io_confidence", "hidden_state", network_ids)

# get the network that has yielded the median decoding performance
median_network_id = decoding_data.get_median_r2_network_id(args.decoder_group_dir, "io_confidence", "hidden_state")
i_network = list(network_ids).index(median_network_id)
network = networks[i_network]
precision_decoder = precision_decoders[i_network]

# get unit axes for the prediction and for the precision orthogonalized with respect to the prediction
w_prediction = network.output_layer.weight.detach().numpy() # shape [1, n_units]
w_precision = precision_decoder.coef_ # shape [1, n_units]
w_precision_orth = (w_precision 
    - ((w_precision @ w_prediction.T) / (w_prediction @ w_prediction.T)) * w_prediction)
w_prediction_unit = w_prediction / np.sqrt(np.sum(w_prediction ** 2))
w_precision_orth_unit = w_precision_orth / np.sqrt(np.sum(w_precision ** 2))

# compute hidden state
hidden_state_by_level = [ _get_hidden_state(network, inputs) for inputs in inputs_by_level ]
# compute 2d state by projecting the hidden state onto the prediction-precision subspace
state_prediction_by_level = [ get_state_component(hidden_state, w_prediction_unit) for hidden_state in hidden_state_by_level ]
state_precision_by_level = [ get_state_component(hidden_state, w_precision_orth_unit) for hidden_state in hidden_state_by_level ]

# Plot
markersize = 1.0
linewidth = 0.5
fig = plt.figure(figsize=figsize, constrained_layout=True)
ax = fig.gca()
for i_level, color in enumerate(COLOR_BY_LEVEL):
    ax.plot(state_prediction_by_level[i_level][T_PLOT_START:],
        state_precision_by_level[i_level][T_PLOT_START:],
        'o--', color=color, ms=markersize, lw=linewidth)
ax.set_xlabel("Prediction")
ax.set_ylabel("Orthognalized precision")
ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax.tick_params(axis='y', which='both', left=False, labelleft=False)

fig.savefig(out_path)
print("Figure saved at", out_path)
