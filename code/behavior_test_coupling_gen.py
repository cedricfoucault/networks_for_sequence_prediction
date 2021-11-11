import torch
import scipy.stats as stats

def create_initial_sequence(rep_item, pre_rep_pgen, pre_other_pgen, t_query_before):
    inputs = torch.empty((t_query_before))
    obs_tminus1 = 0
    for t in range(t_query_before):
        obs_tminus1_is_other = obs_tminus1 != rep_item
        pgen = pre_other_pgen if obs_tminus1_is_other else pre_rep_pgen
        rep_event = stats.bernoulli.rvs(pgen)
        obs_t_is_other = obs_tminus1_is_other == (rep_event == 1)
        inputs[t] = 1 - rep_item if obs_t_is_other else rep_item
        obs_tminus1 = inputs[t]
    return inputs

def _fill_with_repetitions_and_queries(inputs, rep_item, t_query_before, t_query_after):
    inputs[t_query_before, :, 0] = 1 - rep_item # query before the streak
    inputs[t_query_before + 1 : t_query_after, :, 0] = rep_item # repetition streak
    inputs[t_query_after, :, 0] = 1 - rep_item # query after the streak

