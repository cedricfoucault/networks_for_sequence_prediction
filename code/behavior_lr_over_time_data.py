import argparse
import generate as gen
import io_model
import measure
import model
import numpy as np
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
parser.add_argument("group_ids", nargs="+", help="group ids to test")
parser.add_argument("-o", "--output", help="path to output file")
args = parser.parse_args()
group_ids = args.group_ids
out_path = args.output

assert len(group_ids) > 0, "At least one group_id is required"

# Generate reversal sequences
gen_process = training_data.load_gen_process(group_ids[0])
n_time_steps = gen.SEQ_N_TIME_STEPS
n_reversals = round(gen_process.p_change * n_time_steps)
delay_other_reversals = n_time_steps // n_reversals
delay_first_reversal = n_time_steps - 1 - delay_other_reversals * (n_reversals - 1)
reversal_times = np.array([ delay_first_reversal + i * delay_other_reversals for i in range(n_reversals) ])
n_sequences = 10000
p_0 = 0.8
inputs = torch.empty(n_time_steps, n_sequences, 1)
assert reversal_times[-1] == n_time_steps - 1, "programmer error"
for iSeq in range(n_sequences):
    p_gen = p_0
    t = 0
    for t_change in reversal_times:
        inputs[t : t_change + 1, iSeq, 0] = torch.from_numpy(stats.bernoulli.rvs(p_gen, size=t_change + 1 - t))
        p_gen = 1 - p_gen
        t = t_change + 1

# Get learning rates for each model
io = io_model.IOModel(gen_process)
io_predictions = io.get_predictions(inputs)
io_lrs = measure.get_learning_rate(io_predictions, inputs)
io_lr = sequence.get_numpy1D_from_tensor(io_lrs.mean(dim=1)) # average over sequences
io_data_dict = dict(mean=io_lr)
data_by_groupid = {}
for i_group, group_id in enumerate(group_ids):
    models = training_data.load_models(group_id)
    n_models = len(models)
    lrs = np.empty((n_models, n_time_steps))
    for i_model, model_i in enumerate(models):
        predictions = model_i(inputs)
        lrs_s = measure.get_learning_rate(predictions, inputs)
        lrs[i_model, :] = sequence.get_numpy1D_from_tensor(lrs_s.mean(dim=1))
    data_dict = measure.get_summary_stats(lrs, axis=0)
    data_by_groupid[group_id] = data_dict

out_dict = {
    "n_time_steps": n_time_steps,
    "reversal_times": reversal_times,
    "data_io": io_data_dict,
    "data_by_groupid": data_by_groupid
}
torch.save(out_dict, out_path)

