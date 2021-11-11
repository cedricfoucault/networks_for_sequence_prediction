# -*- coding: utf-8 -*-
import os
import skopt
from ray import tune
from ray.tune.suggest.skopt import SkOptSearch

UNIT_TYPE_KEY = "unit_type" # type of hidden layer units
N_UNITS_KEY = "n_units" # number of units in the hidden layer
N_LAYERS_KEY = "n_layers" # number of hidden layers
MODEL_CLASS_KEY = "model_class" # network, deltarule
ESTIMATE_TYPE = "estimate_type" # for deltarule: "item", "transition"
OPTIMIZER_NAME_KEY = "optimizer.name" # optimization algorithm name
OPTIMIZER_LR_KEY = "optimizer.lr" # optimizer learning rate
OPTIMIZER_MOMENTUM_KEY = "optimizer.momentum" # momentum (for RMSprop optimizer)
OPTIMIZER_ETAMINUS_KEY = "optimizer.etaminus" # eta- (for Rprop optimizer only)
OPTIMIZER_ETAPLUS_KEY = "optimizer.etaplus" # eta- (for Rprop optimizer only)
TRAIN_OUTPUT_ONLY_KEY = "train_output_only" # optional boolean flag to train only
# the output layer parameters of a network, freezing the other parameters
INITIALIZATION_SCHEME_KEY = "initialization_scheme"
# optional, to specify an initialization scheme for the weight matrices other than the default.
# possible values: "diagonal"
INIT_DIAGONAL_MEAN_KEY = "init_diagonal_mean"
# optional, to specify the mean of the initial diagonal coefficients of the recurrent weight matrix
# (if unspecified, defaults to 0)
TRAIN_DIAGONAL_ONLY_KEY = "train_diagonal_only"
# optional boolean flag to train only the diagonal coefficients of the recurrent weight matrix of a network
INITIALIZATION_WEIGHT_INPUT_TO_HIDDEN_STD_KEY = "initialization.weight.input_to_hidden.std"
# optional, to specify the standard deviation of the normal distribution
# from which input-to-hidden weights are sampled from at initialization time
# if unspecified, std = 1 / sqrt(N) where N is the number of units
INITIALIZATION_WEIGHT_HIDDEN_TO_HIDDEN_STD_KEY = "initialization.weight.hidden_to_hidden.std"
# optional, to specify the standard deviation of the normal distribution
# from which hidden-to-hidden weights are sampled from at initialization time
# if unspecified, std = 1 / sqrt(N) where N is the number of units
INITIALIZATION_WEIGHT_HIDDEN_TO_OUTPUT_STD_KEY = "initialization.weight.hidden_to_output.std"
# if this option is enabled, during training the binary cross entropy loss
# will be computed using the logits rather than the predictions.
# this may provide better numerical stability.
ENABLE_COMPUTE_LOSS_WITH_LOGITS_DURING_TRAINING = "enable_compute_loss_with_logits_during_training"

def load_config(path):
    with open(path, 'r') as f:
        configdict_as_txt = f.read()
    return eval(configdict_as_txt)


