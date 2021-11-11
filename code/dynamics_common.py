import generate as gen
import torch
import utils

N_TIME_STEPS_PRIOR = 100
N_TIME_STEPS_STREAK = 20
T_PLOT_START = N_TIME_STEPS_PRIOR - 1
PRIOR_P_GEN_LEVELS = [1 / 2., 1 / 3., 1/ 5.]
DEFAULT_SEED = 27
COLOR_BY_LEVEL = ["#2aa02b", "#bdb421", "#e378c1"]

def get_inputs_by_level_with_seed(seed):
    inputs_by_level = []
    for i_level, p_gen in enumerate(PRIOR_P_GEN_LEVELS):
        utils.set_seed(seed)
        inputs, _ = gen.generate_sequence_bernoulli_with_change_times_and_pvals(N_TIME_STEPS_PRIOR, [], [p_gen])
        inputs = inputs.reshape(N_TIME_STEPS_PRIOR, 1, 1)
        inputs_by_level.append(inputs)
    # append the same observations to all input sequences to see the different behaviors as a function of the prior
    end_inputs = torch.ones((N_TIME_STEPS_STREAK, 1, 1), dtype=inputs_by_level[0].dtype)
    inputs_by_level = [ torch.cat((inputs, end_inputs), dim=0) for inputs in inputs_by_level ]
    return inputs_by_level

def print_inputs_by_level(inputs_by_level):
    for i_level, inputs in enumerate(inputs_by_level):
        input_seq_str = "".join([ str(int(x)) for x in inputs.squeeze().detach().numpy() ])
        input_seq_str = input_seq_str[:T_PLOT_START+1] + "•" + input_seq_str[T_PLOT_START+1:]
        print(f"input sequence {i_level + 1}: {input_seq_str}")
