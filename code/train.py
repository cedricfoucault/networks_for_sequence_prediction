import hyperparam_config as hconfig
import model
import numpy as np
import torch

def get_trained_model_with_config(train_data, config,
                                  progress_callback=None,
                                  callback_before_training=False,
                                  progress_step_size=10):
    model_class = config.get(hconfig.MODEL_CLASS_KEY, "network")
    model_instance = model.get_class_from_string(model_class).new_with_config(config)
    train_model_with_config(train_data, model_instance, config,
                            progress_callback,
                            callback_before_training,
                            progress_step_size)
    return model_instance
    

def train_model_with_config(train_data, model_instance, config,
                            progress_callback=None,
                            callback_before_training=False,
                            progress_step_size=10):
    if callback_before_training and (progress_callback is not None):
        progress_callback(model_instance, -1)
    train_output_only = config.get(hconfig.TRAIN_OUTPUT_ONLY_KEY, False)
    if train_output_only:
        assert isinstance(model_instance, model.Network), \
            f"wrong model class for use with {hconfig.TRAIN_OUTPUT_ONLY_KEY}"
        # freeze model parameters
        for param in model_instance.parameters():
            param.requires_grad = False
        # unfreeze output layer parameters and pass only them to the optimizer
        for param in model_instance.output_layer.parameters():
            param.requires_grad = True
        parameters = model_instance.output_layer.parameters()
    else:
        parameters = model_instance.parameters()
    train_diagonal_only = config.get(hconfig.TRAIN_DIAGONAL_ONLY_KEY, False)
    enable_compute_loss_with_logits = config.get(
        hconfig.ENABLE_COMPUTE_LOSS_WITH_LOGITS_DURING_TRAINING,
        False)
    optimizer = get_optimizer_with_config(parameters, config)
    train_network(train_data, model_instance, optimizer,
        progress_callback=progress_callback, progress_step_size=progress_step_size,
        train_diagonal_only=train_diagonal_only,
        enable_compute_loss_with_logits=enable_compute_loss_with_logits)

def get_optimizer_with_config(parameters, config):
    optimizer_name = config[hconfig.OPTIMIZER_NAME_KEY]
    optimizer_params = {}
    if hconfig.OPTIMIZER_LR_KEY in config:
        optimizer_params["lr"] = config[hconfig.OPTIMIZER_LR_KEY]
    if hconfig.OPTIMIZER_MOMENTUM_KEY in config:
        optimizer_params["momentum"] = config[hconfig.OPTIMIZER_MOMENTUM_KEY]
    if (hconfig.OPTIMIZER_ETAMINUS_KEY in config) and \
        (hconfig.OPTIMIZER_ETAPLUS_KEY in config):
        optimizer_params["etas"] = [ config[hconfig.OPTIMIZER_ETAMINUS_KEY],
                         config[hconfig.OPTIMIZER_ETAPLUS_KEY] ]
    optimizer_function = getattr(torch.optim, optimizer_name)
    return optimizer_function(parameters, **optimizer_params)

def train_network(train_data, network, optimizer, progress_callback=None, progress_step_size=10,
    train_diagonal_only=False, enable_compute_loss_with_logits=False):
    train_minibatches_inputs, train_minibatches_targets, _ = train_data
    train_n_minibatches = len(train_minibatches_inputs)
    # Training loop
    network.train(True) # Set the network in training mode
    for iMinibatch in range(train_n_minibatches):
        # get network input and target for this minibatch
        inputs = train_minibatches_inputs[iMinibatch]
        targets = train_minibatches_targets[iMinibatch]
        # compute the loss between output and target
        if enable_compute_loss_with_logits:
            logits = network.forward_logits(inputs)
            loss = torch.nn.BCEWithLogitsLoss()(logits, targets)
        else:
            outputs = network(inputs)
            loss = torch.nn.BCELoss()(outputs, targets)
        # zero the gradients
        optimizer.zero_grad()
        # compute the gradients through backpropagation
        loss.backward()
        if train_diagonal_only:
            network._zero_nondiagonal_weights_gradient()
        # update the weights
        optimizer.step()
        # progress callback
        if (progress_callback is not None) and \
           iMinibatch % progress_step_size == 0:
           progress_callback(network, iMinibatch)
        
    if (progress_callback is not None):
        progress_callback(network, train_n_minibatches-1)

