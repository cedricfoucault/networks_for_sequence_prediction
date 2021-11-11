# Behavioral test for measuring the effect of
# being trained on coupled vs independent environments
# on the predictions
# (whether networks generalize the fact that one transition probability has changed
# to the other transition probability)

# Method:
# - Create sequences with an initial period of low repetition probability for one of the two items,
#   and then a period of repetitions of that item
# - Measure a model's prediction for the transition probability that is conditional
#   on the other item, pre and post the repetition period
# - Hypothesis: if a model believes the probabilities are coupled,
#   it should have a greater reset (absolute pre-post difference) of prediction
#   than if it believes they are independent

# This script computes the data for this test

# /!\ Note: this is insanely long to run.
# If this is to be run multiple times in the future, consider how to speed this up.

import argparse
from behavior_test_coupling_gen import create_initial_sequence, _fill_with_repetitions_and_queries
import io_model
import numpy as np
import pandas
import scipy.stats as stats
import torch
import training_data
import utils

verbose = False
seed = 34
utils.set_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("group_id_coupled", type=str, help="group id of coupled network models")
parser.add_argument("group_id_independent", type=str, help="group id of independent network models")
utils.add_arguments(parser, ["output"])
parser.add_argument('--test_ideal_observer', dest='test_ideal_observer', action='store_true',
    help="""test the ideal observers instead of the networks.
    the group ids are still used to know the generative processes of the ideal observers.""")
parser.set_defaults(test_ideal_observer=False)
args = parser.parse_args()
group_id_coupled = args.group_id_coupled
group_id_independent = args.group_id_independent
out_path = args.output
test_ideal_observer = args.test_ideal_observer

gen_process_coupled = training_data.load_gen_process(group_id_coupled)
gen_process_independent = training_data.load_gen_process(group_id_independent)
if test_ideal_observer:
    io_coupled = io_model.IOModel(gen_process_coupled)
    io_independent = io_model.IOModel(gen_process_independent)
    models_by_coupled = {
        False: [io_independent],
        True: [io_coupled]
    }
    n_models = 2
else:
    networks_coupled, network_ids_coupled = training_data.load_models_ids(group_id_coupled)
    networks_independent, network_ids_independent = training_data.load_models_ids(group_id_independent)
    models_by_coupled = {
        False: networks_independent,
        True: networks_coupled
    }
    n_models = len(networks_coupled) + len(networks_independent)

## Create sequences data
low_probability = 0.2
pre_rep_pgen = low_probability # initial repetition probability
n_sequences = 100
# t_query_before:
# time at which the prediction for the unobserved bigram is queried
# just before the repetition streak
t_query_before = 74
rep_lengths = range(0, 76) # possible values for the streak length (number of repetitions in the streak)

# There are 4 quadrants for 2*2 possibilities:
# - whether the repeating item is 0 or 1
# - whether the transition probability conditional on the non-repeating item
#   is initially high or low
n_quadrants = 4
n_rep_lengths = len(rep_lengths)

size = (n_quadrants, n_rep_lengths, n_models)
df_rep_item = np.empty(size)
df_pre_rep_pgen = np.empty(size)
df_pre_other_pgen = np.empty(size)
df_rep_length = np.empty(size)
df_is_coupled = np.empty(size, dtype=bool)

df_p_after = np.empty((n_quadrants, n_rep_lengths, n_models, n_sequences))
df_p_before = np.empty((n_quadrants, n_rep_lengths, n_models, n_sequences))

for quadrant in range(n_quadrants):
    if verbose:
        print("quadrant", quadrant)
    rep_item = quadrant // 2
    pre_other_pgen = 1 - low_probability if quadrant % 2 == 0 \
        else low_probability
    
    max_input_length = t_query_before + 1 + max(rep_lengths) + 1
    inputs = torch.zeros((max_input_length, n_sequences, 1))
    for i_seq in range(n_sequences):
        initial_sequence = create_initial_sequence(rep_item, pre_rep_pgen, pre_other_pgen, t_query_before)
        inputs[0 : t_query_before, i_seq, 0] = initial_sequence
        
    for i_rep_length, rep_length in enumerate(rep_lengths):
        if verbose:
            print("rep_length", rep_length)
        # t_query_after:
        # time at which the prediction for the unobserved bigram is queried
        # right after the repetition streak
        t_query_after = t_query_before + 1 + rep_length
        _fill_with_repetitions_and_queries(inputs, rep_item, t_query_before, t_query_after)
        
        i_model = 0
        for is_coupled in [False, True]:
            models = models_by_coupled[is_coupled]
            for model in models:
                df_rep_item[quadrant, i_rep_length, i_model] = rep_item
                df_pre_rep_pgen[quadrant, i_rep_length, i_model] = pre_rep_pgen
                df_pre_other_pgen[quadrant, i_rep_length, i_model] = pre_other_pgen
                df_rep_length[quadrant, i_rep_length, i_model] = rep_length
                df_is_coupled[quadrant, i_rep_length, i_model] = is_coupled
                
                predictions = model(inputs)
                df_p_before[quadrant, i_rep_length, i_model, :] = \
                    predictions[t_query_before, :, 0].detach().numpy()
                df_p_after[quadrant, i_rep_length, i_model, :] = \
                    predictions[t_query_after, :, 0].detach().numpy()

                i_model += 1

# Average over sequences
df_p_after = df_p_after.mean(axis=3)
df_p_before = df_p_before.mean(axis=3)

df_dict = {
    "group_id_coupled": group_id_coupled,
    "group_id_independent": group_id_independent,
    "rep_item": df_rep_item.flatten(),
    "pre_rep_pgen": df_pre_rep_pgen.flatten(),
    "pre_other_pgen": df_pre_other_pgen.flatten(),
    "t_query_before": t_query_before,
    "rep_length": df_rep_length.flatten(),
    "is_coupled": df_is_coupled.flatten(),
    "p_before": df_p_before.flatten(),
    "p_after": df_p_after.flatten(),
}
df = pandas.DataFrame(df_dict)
df.to_csv(out_path)
print("data saved at", out_path)
        


