import argparse
import data
import measure
from model import get_new_model_instance_with_config
import numpy as np
import os
import utils
import hyperparam_config as hconfig
import ray
from ray import tune
import threading
import torch
import train

# Parse Input
parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str, help="path to the hyperparam config dict file", metavar="hyperparam_config_files/config123.txt")
parser.add_argument("-o", "--output", help="path to output directory")
args = parser.parse_args()
config_path = args.config_path
out_path = args.output

# # Output path
if not out_path:
    assert False, "argument required: -o"

# Hyperparameters
tuneconfig = hconfig.load_config(config_path)
progress_step_size = tuneconfig.get("progress_step_size", 10)
shuffle_minibatches = tuneconfig.get("shuffle_minibatches", True)
no_train = tuneconfig.get("no_train", False)
save_parameters_through_training = tuneconfig.get("save_parameters_through_training", False)
num_cpus = tuneconfig.get("num_cpus", None)
do_save_model = tuneconfig.get("do_save_model", True)
do_catch_error = tuneconfig.get("do_catch_error", False)

# set trial name manually with a global trial counter,
# in order to then retrieve it and generate a unique seed per trial
trial_count = 0
def fun_trial_name_creator(config):
    global trial_count
    trial_name = str(trial_count)
    trial_count += 1
    return trial_name

def run_trial(tuneconfig):
    # set unique seed for this trial for reproducibility
    trial_name = tune.get_trial_name()
    trial_index = int(trial_name)
    trial_seed = tuneconfig["seed"] + trial_index
    utils.set_seed(trial_seed)

    try:
        if no_train:
            model = get_new_model_instance_with_config(tuneconfig)
            tune.report(validation_loss=float("inf"), n_training_updates=0)
        else:
            # Load training data
            train_dataset_name = tuneconfig["train_dataset_name"]
            validate_dataset_name = tuneconfig["validate_dataset_name"]
            train_data, gen_process = data.load_train_data(train_dataset_name)
            if shuffle_minibatches:
                input_minibatches, target_minibatches, p_gen_minibatches = train_data
                perm = np.random.permutation(len(input_minibatches))
                input_minibatches = [ input_minibatches[i] for i in perm ]
                target_minibatches = [ target_minibatches[i] for i in perm ]
                p_gen_minibatches = [ p_gen_minibatches[i] for i in perm ]
                train_data = input_minibatches, target_minibatches, p_gen_minibatches
            validate_data = data.load_test_data(validate_dataset_name)
            validate_inputs, validate_targets, _ = validate_data
            if save_parameters_through_training:
                parameters_through_training = {}
            def progress_callback_fun(network, iMinibatch):
                n_training_updates = iMinibatch + 1
                if save_parameters_through_training:
                    parameters_through_training[n_training_updates] = network.extract_parameters_dict()
                # Measure loss on validation data
                network.train(False)
                validate_outputs = network(validate_inputs)
                validation_loss = measure.get_loss(validate_outputs, validate_targets)
                tune.report(validation_loss=validation_loss, n_training_updates=n_training_updates)
                network.train(True)
            # Train models with given hyperparameters
            model = train.get_trained_model_with_config(train_data, tuneconfig,
                                                          progress_callback=progress_callback_fun,
                                                          callback_before_training=True,
                                                          progress_step_size=progress_step_size)
        model.train(False)
        if do_save_model:
            model.save("trained_model.pt")
        if save_parameters_through_training:
            torch.save(parameters_through_training, "parameters_through_training.pt")

    except RuntimeError:
        if not do_catch_error:
            raise
        bad_loss = measure.get_loss(torch.zeros_like(validate_targets) + 1e-3, validate_targets)
        tune.report(validation_loss=bad_loss, n_training_updates=-1)
    

# Boot ray engine
ray.shutdown()
ray.init(num_cpus=num_cpus)

# Initialize seed
seed = tuneconfig["seed"]
utils.set_seed(seed)

# Run trials with given configuration
local_dir = os.getcwd()
analysis = tune.run(
    run_trial, 
    config=tuneconfig, 
    verbose=1,
    name=out_path,
    num_samples=tuneconfig.get("num_samples", 1),
    search_alg=tuneconfig.get("search_alg", None),
    scheduler=tuneconfig.get("scheduler", None),
    local_dir=local_dir,
    trial_name_creator=fun_trial_name_creator,
    fail_fast=tuneconfig.get("fail_fast", True),
    metric="validation_loss",
    mode="min",
)

print("all data saved at ", out_path)

