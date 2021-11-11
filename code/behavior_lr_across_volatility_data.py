import argparse
import data
import io_model
import measure
import numpy as np
import os
import torch
import training_data


volatility_denominator = 300

parser = argparse.ArgumentParser()
parser.add_argument("test_dataset_names", type=str, nargs='+', help="list of test datasets")
parser.add_argument("--group_ids", type=str, nargs='+', help="list of trained model group ids")
parser.add_argument("-o", "--output", help="path to output for indidvidual data")
args = parser.parse_args()
test_dataset_names = args.test_dataset_names
network_group_ids = args.group_ids
out_path = args.output

n_train = len(network_group_ids)
n_test = len(test_dataset_names)
volatility_train_numerators = np.empty((n_train), dtype=int)
volatility_test_numerators = np.empty((n_test), dtype=int)
network_groups = []
ios = []
lr_mean_matrix = np.empty((n_train, n_test))
lr_io_matrix = np.empty((n_train, n_test))
for i_train, group_id in enumerate(network_group_ids):
    gen_process = training_data.load_gen_process(group_id)
    volatility_train_numerators[i_train] = round(volatility_denominator * gen_process.p_change)
    networks = training_data.load_models(group_id)
    network_groups.append(networks)
    io = io_model.IOModel(gen_process)
    ios.append(io)
for i_test, test_dataset_name in enumerate(test_dataset_names):
    inputs, targets, p_gens = data.load_test_data(test_dataset_name)
    gen_process = data.load_gen_process(test_dataset_name)
    volatility_test_numerators[i_test] = round(volatility_denominator * gen_process.p_change)
    for i_train, networks in enumerate(network_groups):
        lr_means = np.empty(len(networks))
        for i_net, network in enumerate(networks):
            predictions = network(inputs)
            lrs = measure.get_learning_rate(predictions, inputs)
            lr_means[i_net] = lrs.mean().item()
        lr_mean_matrix[i_train, i_test] = lr_means.mean()
        io = ios[i_train]
        io_predictions = io.get_predictions(inputs)
        lr_io_matrix[i_train, i_test] = measure.get_learning_rate(io_predictions, inputs).mean()

out_dict = dict(
    volatility_denominator=volatility_denominator,
    volatility_train_numerators=volatility_train_numerators,
    volatility_test_numerators=volatility_test_numerators,
    lr_mean_matrix=lr_mean_matrix,
    lr_io_matrix=lr_io_matrix
)
torch.save(out_dict, out_path)
print("Data saved at", out_path)
