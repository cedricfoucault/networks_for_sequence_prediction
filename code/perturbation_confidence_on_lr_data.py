import argparse
import data
import decoding_data
import numpy as np
import measure
import os
import pandas
from perturbation_confidence_module import generate_random_perturbation_vector, get_perturbation_prototypes, get_perturbation_norm
import torch
import training_data
import utils

#### Compute effect of perturbation that provokes a change confidence at time t
#### without changing the prediction at time t
#### on the apparent learning rate at time t+1
#### (i.e. the change in prediction at time t+1 normalized by the prediction error)

seed = 65
utils.set_seed(65)

parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str, help="name of test dataset")
parser.add_argument("group_id", type=str, help="trained model group id")
parser.add_argument("decoder_group_dir", type=str, help="path to decoder group directory")
parser.add_argument("-o", "--output", help="path to output file")
args = parser.parse_args()
dataset_name = args.dataset_name
group_id = args.group_id
decoder_group_dir = args.decoder_group_dir
out_path = args.output

n_data_samples = int(1e3)


sanity_check = False
epsilon = 1e-3

outcome_key = "io_confidence"
predictor_key = "hidden_state"
networks, network_ids = training_data.load_models_ids(group_id)
confidence_decoders = decoding_data.load_decoders(decoder_group_dir, outcome_key, predictor_key, network_ids)
n_networks = len(networks)

# Desired confidence change values
min_conf = 0. # empirically determined
max_conf = 3.0 # empirically determined
delta_conf_half_range = (max_conf - min_conf) / 6.
n_delta_confs = 5
delta_confs = np.linspace(-delta_conf_half_range, +delta_conf_half_range, num=n_delta_confs)
max_delta_conf = np.absolute(delta_confs).max()
max_confidence_ratio = 0.9

# Pre-compute perturbation prototypes
prototypes = get_perturbation_prototypes(networks, confidence_decoders)
# perturbation_norm = get_perturbation_fixed_norm(prototypes, np.absolute(delta_confs).max())

# Pre-compute inputs and hidden states
inputs, target, p_gens = data.load_test_data(dataset_name)
hidden_states_s = [ network.forward_activations(inputs)["hidden_state"] for network in networks ]

def get_lr_tplus1(outputs_t, inputs_tplus1, outputs_tplus1):
    return (outputs_tplus1 - outputs_t) / (inputs_tplus1 - outputs_t)

n_time_steps = inputs.size(0)
n_sequences = inputs.size(1)
# Sample n_data_samples time and sequence indices
ts = np.random.randint(n_time_steps - 1, size=n_data_samples)
i_seqs = np.random.randint(n_sequences, size=n_data_samples)
# Compute delta_lr for each network, and delta_conf level, and data sample
delta_lrs = np.empty((n_delta_confs, n_networks, n_data_samples))
for i_network, network in enumerate(networks):
    decoder = confidence_decoders[i_network]
    hidden_states = hidden_states_s[i_network]
    prototype = prototypes[i_network]
    perturbation_norm = get_perturbation_norm(prototype, max_delta_conf, max_confidence_ratio)
    # print("calculating delta_lrs for network {:d}".format(i_network))
    for i_delta, delta_conf in enumerate(delta_confs):
        for i_sample in range(n_data_samples):
            perturbation_vector = generate_random_perturbation_vector(prototype, perturbation_norm, delta_conf)
            perturbation = torch.tensor(perturbation_vector, dtype=hidden_states.dtype)
            t = ts[i_sample]
            i_seq = i_seqs[i_sample]
            inputs_tplus1 = inputs[t + 1, i_seq:(i_seq+1), :]
            hidden_state_t = hidden_states[t, i_seq:(i_seq+1), :]
            outputs_t = network.compute_outputs_from_hidden_state(hidden_state_t)
            outputs_tplus1 = network.compute_next_activations(inputs_tplus1, hidden_state_t)["outputs"]
            outputs_tplus1_perturbed = network.compute_next_activations(inputs_tplus1, hidden_state_t + perturbation)["outputs"]
            lr_tplus1 = get_lr_tplus1(outputs_t, inputs_tplus1, outputs_tplus1)
            lr_tplus1_perturbed = get_lr_tplus1(outputs_t, inputs_tplus1, outputs_tplus1_perturbed)
            delta_lrs[i_delta, i_network, i_sample] = (lr_tplus1_perturbed - lr_tplus1).squeeze().item()


np.savez(out_path, delta_confs=delta_confs, delta_lrs=delta_lrs)
print("Data saved at", out_path)



