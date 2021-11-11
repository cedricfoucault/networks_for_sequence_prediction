import io_model
import numpy as np
from scipy import stats
import sequence
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import torch

def get_loss(predictions, targets):
    loss_function = torch.nn.BCELoss() # binary cross entropy
    return loss_function(predictions, targets).item()

def get_learning_rate(predictions, inputs):
    learning_rate = torch.empty_like(predictions)
    n_time_steps = learning_rate.size(0)
    prediction_prior = 0.5
    learning_rate[0] = (predictions[0] - prediction_prior) / (inputs[0] - prediction_prior)
    learning_rate[1:n_time_steps] = (predictions[1:n_time_steps] - predictions[0:n_time_steps-1]) / (inputs[1:n_time_steps] - predictions[0:n_time_steps-1])
    return learning_rate

def get_confidence_from_sds(sds):
    return -torch.log(sds)

def get_sds_from_confidence(confidence):
    return torch.exp(-confidence)

def get_logits_from_ps(ps):
    # logit = log-odds function
    return torch.log(ps / (1. - ps))

def get_ps_from_logits(logits):
    return logits.sigmoid()

def get_chance_loss(inputs, targets):
    constant_outputs = torch.ones_like(inputs) * 0.5
    return get_loss(constant_outputs, targets)

def get_io_loss(inputs, targets, gen_process):
    io = io_model.IOModel(gen_process)
    io_predictions = io.get_predictions(inputs)
    return get_loss(io_predictions, targets)

def get_percent_value(values, ref_0, ref_100):
    return (values - ref_0) / (ref_100 - ref_0) * 100.

def get_summary_stats(values, axis=0, ci_threshold=0.95):
    n_values = values.shape[axis]
    mean = values.mean(axis=axis)
    sem = stats.sem(values, axis=axis)
    ci_lower, ci_upper = \
        stats.t.interval(ci_threshold, n_values - 1, loc=mean, scale=sem)
    if np.any(sem == 0):
        ci_lower[sem == 0] = mean[sem == 0]
        ci_upper[sem == 0] = mean[sem == 0]
    return dict(mean=mean, sem=sem, ci_lower=ci_lower, ci_upper=ci_upper)

def get_median_ci(values, ci_threshold=0.95):
    # reference: https://github.com/minddrummer/median-confidence-interval/blob/master/Median_CI.py
    # flat to one dimension array
    assert len(values.shape) == 1, "need 1D array"
    N = values.shape[0]
    data = np.sort(values)
    low_count, up_count = stats.binom.interval(ci_threshold, N, 0.5, loc=0)
    low_count -= 1 # indexing starts at 0
    return data[int(low_count)], data[int(up_count)]

def stats_twoway_anova_table(data, iv1_name, iv2_name, dv_name, type=1, test="F"):
    # Fit a two-way ANOVA model on the data
    anova_formula = f"{dv_name} ~ C({iv1_name}) + C({iv2_name}) + C({iv1_name}):C({iv2_name})"
    anova_model = ols(anova_formula, data=data).fit()
    return anova_lm(anova_model, test=test, typ=type)

def cat_features(tensors):
    return torch.cat(tensors, dim=2)

def get_vector(tensor):
    """
    - input: torch.tensor(n_time_steps, n_samples, n_features) 
    - output: np.ndarray(n_time_steps * n_samples, n_features)
    where:
    output[0:n_samples, :] = input[0, 0:n_samples, :],
    output[n_samples:2*n_samples,:] = input[1, 0:n_samples, :],
    ..."""
    n_features = tensor.size(2) if len(tensor.shape) > 2 else 1
    return tensor.reshape((-1, n_features)).numpy()

def get_tensor(vector, n_time_steps, n_samples):
    """
    Inverse function of get_vector()
    - input: np.ndarray(n_time_steps * n_samples, n_features)
    - output: torch.tensor(n_time_steps, n_samples, n_features) 
    where:
    output[0, 0:n_samples, :] = input[0:n_samples, :],
    output[1, 0:n_samples, :] = input[n_samples:2*n_samples,:],
    ..."""
    assert n_time_steps * n_samples == vector.shape[0], "Inconsistent sizes"
    n_features = vector.shape[1] if len(vector.shape) > 1 else 1
    return torch.tensor(vector).reshape(n_time_steps, n_samples, n_features)

def get_decoded_tensor(decoder, predictor_tensor):
    n_time_steps, n_samples = predictor_tensor.size(0), predictor_tensor.size(1)
    decoded_vector = decoder.predict(get_vector(predictor_tensor))
    return get_tensor(decoded_vector, n_time_steps, n_samples)

def get_predictor_tensor_dict_from_activations(a):
    # we include as predictors
    # - hidden state at time t
    # - hidden gates at time t+1
    # - both (hidden state at time t, gates at time t+1)
    # - output of network at time t (as a control)
    # this requires discarding one time step:
    # the first one for state, the last for gates
    predictor_tensor_dict = {}
    has_state = "hidden_state" in a
    has_gates = "reset_gate" in a and "update_gate" in a
    if has_state:
        state_t = a["hidden_state"][:-1, :, :]
        predictor_tensor_dict["hidden_state"] = state_t
    if has_gates:
        gates_tplus1 = cat_features((a["reset_gate"], a["update_gate"]))[1:, :, :]
        predictor_tensor_dict["hidden_gates_tplus1"] = gates_tplus1
    if has_state and has_gates:
        hidden_all = cat_features((state_t, gates_tplus1))
        predictor_tensor_dict["hidden_all"] = hidden_all
    if "outputs" in a:
        output_t = a["outputs"][:-1, :, :].detach()
        predictor_tensor_dict["output"] = output_t
    return predictor_tensor_dict

def get_predictor_vector_dict_from_activations(a):
    predictor_tensor_dict = get_predictor_tensor_dict_from_activations(a)
    predictor_vector_dict = {
        key: get_vector(tensor) for key, tensor in predictor_tensor_dict.items()
        }
    return predictor_vector_dict

def get_decoder_target_vector_dict_from_io_outputs_inputs(io_outputs, inputs, gen_process):
    decoder_target_tensor_dict = \
        get_decoder_target_tensor_dict_from_io_outputs_inputs(io_outputs, inputs, gen_process)
    decoder_target_vector_dict = {
        key: get_vector(tensor) for key, tensor in decoder_target_tensor_dict.items()
        }
    return decoder_target_vector_dict

def get_decoder_target_tensor_dict_from_io_outputs_inputs(io_outputs, inputs, gen_process):
    # we include as decoder targets:
    # - bayes confidence at time t
    # - bayes evidence at time t
    # - bayes apparent learning rate at time t+1
    # this requires discarding one time step:
    # the first one for confidence and evidence, the last for learning rate
    io_ps = io_outputs["mean"]
    io_sds = io_outputs["sd"]
    if gen_process.is_markov():
        io_confidence = get_confidence_from_sds(io_sds)[:-1, :, :]
        io_evidence = get_logits_from_ps(io_ps)[:-1, :, :]
        io_lr_tplus1 = get_learning_rate(io_ps, inputs)[1:, :, :]
        io_confidence_0 = io_confidence[:, :, 0]
        io_confidence_1 = io_confidence[:, :, 1]
        io_evidence_0 = io_evidence[:, :, 0]
        io_evidence_1 = io_evidence[:, :, 1]
        io_lr_tplus1_0 = io_lr_tplus1[:, :, 0]
        io_lr_tplus1_1 = io_lr_tplus1[:, :, 1]
        # the "relevant" estimate is the one that is conditional on the input at time t
        # (i.e. the estimate that is used for prediction at time t and then is updated at time t+1)
        conditional_inputs = inputs[:-1, :, :]
        io_confidence_xt = sequence.get_relevant_estimates_given_inputs(io_confidence, conditional_inputs)
        io_confidence_1minusxt = sequence.get_relevant_estimates_given_inputs(io_confidence, 1 - conditional_inputs)
        io_evidence_xt = sequence.get_relevant_estimates_given_inputs(io_evidence, conditional_inputs)
        io_evidence_1minusxt = sequence.get_relevant_estimates_given_inputs(io_evidence, 1 - conditional_inputs)
        io_lr_tplus1_xt = sequence.get_relevant_estimates_given_inputs(io_lr_tplus1, conditional_inputs)
        io_lr_tplus1_1minusxt = sequence.get_relevant_estimates_given_inputs(io_lr_tplus1, 1 - conditional_inputs)
        decoder_target_tensor_dict = dict(
            io_confidence_0=io_confidence_0,
            io_confidence_1=io_confidence_1,
            io_evidence_0=io_evidence_0,
            io_evidence_1=io_evidence_1,
            io_lr_tplus1_0=io_lr_tplus1_0,
            io_lr_tplus1_1=io_lr_tplus1_1,
            io_confidence_xt=io_confidence_xt,
            io_confidence_1minusxt=io_confidence_1minusxt,
            io_evidence_xt=io_evidence_xt,
            io_evidence_1minusxt=io_evidence_1minusxt,
            io_lr_tplus1_xt=io_lr_tplus1_xt,
            io_lr_tplus1_1minusxt=io_lr_tplus1_1minusxt
            )
    else:
        io_confidence = get_confidence_from_sds(io_sds)[:-1, :, :]
        io_evidence = get_logits_from_ps(io_ps)[:-1, :, :]
        io_lr_tplus1 = get_learning_rate(io_ps, inputs)[1:, :, :]
        decoder_target_tensor_dict = dict(io_confidence=io_confidence,
                                   io_evidence=io_evidence,
                                   io_lr_tplus1=io_lr_tplus1)
    return decoder_target_tensor_dict
