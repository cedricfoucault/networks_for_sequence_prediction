import numpy as np
import sequence
import scipy.stats as stats
import torch

PROCESS_TYPE_BERNOULLI_FIXED_CHANGE_POINTS = 1 # Bernoulli process with fixed change points
PROCESS_TYPE_BERNOULLI_RANDOM_CHANGE_POINTS = 2 # Bernoulli process with random change points
PROCESS_TYPE_MARKOV_COUPLED_RANDOM_CHANGE_POINTS = 3 # Markov process with random change points where transition probabilities are coupled (i.e. they change at the same time)
PROCESS_TYPE_BERNOULLI_RANDOM_MULTIPLE_CHANGE_POINTS = 4 # multiple volatility levels
PROCESS_TYPE_MARKOV_HEILBRON_MEYNIEL_2019 = 5 # see Heilbron & Meyniel (2019) for exact specs
PROCESS_TYPE_MARKOV_INDEPENDENT_RANDOM_CHANGE_POINTS = 6 # Markov process with random and independent change points
PROCESS_TYPE_ALTREP_COUPLED = 7 # Alternations/Repetitions process with random and coupled change points

## Sequence Data Generation constants
SEQ_N_TIME_STEPS = 380
P_GEN_RANGE_DEFAULT = (0., 1.) # the generative probabilities are sampled uniformly in this range

class GenerativeProcess(object):
    """Parameters of the process for generating sequences of observations"""
    def __init__(self, process_type, p_gen_range=P_GEN_RANGE_DEFAULT):
        super(GenerativeProcess, self).__init__()
        self.type = process_type
        self.p_gen_range = p_gen_range
    
    def is_markov(self):
        return self.is_markov_coupled() or self.is_markov_independent()
    
    def is_markov_coupled(self):
        return self.type == PROCESS_TYPE_MARKOV_COUPLED_RANDOM_CHANGE_POINTS or \
               self.type == PROCESS_TYPE_MARKOV_HEILBRON_MEYNIEL_2019 or \
               self.type == PROCESS_TYPE_ALTREP_COUPLED
    
    def is_markov_independent(self):
        return self.type == PROCESS_TYPE_MARKOV_INDEPENDENT_RANDOM_CHANGE_POINTS
    
    def get_label(self, abbreviated=False):
        if self.is_markov_coupled():
            return "ch. bigram coup." if abbreviated else "changing bigram w. coupled change points"
        elif self.is_markov_independent():
            return "ch. bigram ind." if abbreviated else "changing bigram w. independent change points"
        else:
            return "ch. unigram" if abbreviated else "changing unigram"

    def has_equal_type(self, other):
        return self.type == other.type
        

class GenerativeProcessBernoulliFixed(GenerativeProcess):
    def __init__(self, n_change_points, p_gen_range=P_GEN_RANGE_DEFAULT):
        super(GenerativeProcessBernoulliFixed, self).__init__(PROCESS_TYPE_BERNOULLI_FIXED_CHANGE_POINTS, p_gen_range)
        self.n_change_points = n_change_points
        
class GenerativeProcessBernoulliRandom(GenerativeProcess):
    def __init__(self, p_change, p_gen_range=P_GEN_RANGE_DEFAULT):
        super(GenerativeProcessBernoulliRandom, self).__init__(PROCESS_TYPE_BERNOULLI_RANDOM_CHANGE_POINTS, p_gen_range)
        self.p_change = p_change

class GenerativeProcessMarkovCoupled(GenerativeProcess):
    def __init__(self, p_change, p_gen_range=P_GEN_RANGE_DEFAULT):
        super(GenerativeProcessMarkovCoupled, self).__init__(PROCESS_TYPE_MARKOV_COUPLED_RANDOM_CHANGE_POINTS, p_gen_range)
        self.p_change = p_change

class GenerativeProcessMarkovIndependent(GenerativeProcess):
    def __init__(self, p_change, p_gen_range=P_GEN_RANGE_DEFAULT):
        super(GenerativeProcessMarkovIndependent, self).__init__(PROCESS_TYPE_MARKOV_INDEPENDENT_RANDOM_CHANGE_POINTS, p_gen_range)
        self.p_change = p_change

class GenerativeProcessBernoulliRandomMultiple(GenerativeProcess):
    def __init__(self, p_change_list, p_gen_range=P_GEN_RANGE_DEFAULT):
        super(GenerativeProcessBernoulliRandomMultiple, self).__init__(PROCESS_TYPE_BERNOULLI_RANDOM_MULTIPLE_CHANGE_POINTS, p_gen_range)
        self.p_change_list = p_change_list

class GenerativeProcessHeilbronMeyniel2019(GenerativeProcess):
    def __init__(self, p_change=1./75., p_gen_range=(0.1, 0.9)):
        super(GenerativeProcessHeilbronMeyniel2019, self).__init__(PROCESS_TYPE_MARKOV_HEILBRON_MEYNIEL_2019, p_gen_range)
        self.p_change = p_change
        
class GenerativeProcessAltRepCoupled(GenerativeProcess):
    def __init__(self, p_change, p_gen_range=P_GEN_RANGE_DEFAULT):
        super(GenerativeProcessAltRepCoupled, self).__init__(PROCESS_TYPE_ALTREP_COUPLED, p_gen_range)
        self.p_change = p_change

def get_gen_process_class_from_type(process_type):
    if process_type == PROCESS_TYPE_BERNOULLI_FIXED_CHANGE_POINTS:
        return GenerativeProcessBernoulliFixed
    elif process_type == PROCESS_TYPE_BERNOULLI_RANDOM_CHANGE_POINTS:
        return GenerativeProcessBernoulliRandom
    elif process_type == PROCESS_TYPE_MARKOV_COUPLED_RANDOM_CHANGE_POINTS:
        return GenerativeProcessMarkovCoupled
    elif process_type == PROCESS_TYPE_MARKOV_INDEPENDENT_RANDOM_CHANGE_POINTS:
        return GenerativeProcessMarkovIndependent
    elif process_type == PROCESS_TYPE_BERNOULLI_RANDOM_MULTIPLE_CHANGE_POINTS:
        return GenerativeProcessBernoulliRandomMultiple
    elif process_type == PROCESS_TYPE_MARKOV_HEILBRON_MEYNIEL_2019:
        return GenerativeProcessHeilbronMeyniel2019
    elif process_type == PROCESS_TYPE_ALTREP_COUPLED:
        return GenerativeProcessAltRepCoupled
    else:
        assert False, "unknown type"

def get_dict_from_gen_process(gen_process):
    gen_process_dict = {
            "type": gen_process.type,
            "p_gen_range": gen_process.p_gen_range
            }
    if gen_process.type == PROCESS_TYPE_BERNOULLI_RANDOM_CHANGE_POINTS or \
       gen_process.type == PROCESS_TYPE_MARKOV_COUPLED_RANDOM_CHANGE_POINTS or \
       gen_process.type == PROCESS_TYPE_MARKOV_INDEPENDENT_RANDOM_CHANGE_POINTS or \
       gen_process.type == PROCESS_TYPE_MARKOV_HEILBRON_MEYNIEL_2019 or \
       gen_process.type == PROCESS_TYPE_ALTREP_COUPLED:
        gen_process_dict["p_change"] = gen_process.p_change
    elif gen_process.type == PROCESS_TYPE_BERNOULLI_RANDOM_MULTIPLE_CHANGE_POINTS:
        gen_process_dict["p_change_list"] = gen_process.p_change_list
    elif gen_process.type == PROCESS_TYPE_BERNOULLI_FIXED_CHANGE_POINTS:
        gen_process_dict["n_change_points"] = gen_process.n_change_points
    else:
        assert False, "unknown type"
    return gen_process_dict

def get_gen_process_from_dict(gen_process_dict):
    gen_process_type = gen_process_dict["type"]
    p_gen_range = gen_process_dict["p_gen_range"]
    klass = get_gen_process_class_from_type(gen_process_type)
    if gen_process_type == PROCESS_TYPE_BERNOULLI_RANDOM_CHANGE_POINTS or \
       gen_process_type == PROCESS_TYPE_MARKOV_COUPLED_RANDOM_CHANGE_POINTS or \
       gen_process_type == PROCESS_TYPE_MARKOV_INDEPENDENT_RANDOM_CHANGE_POINTS or \
       gen_process_type == PROCESS_TYPE_MARKOV_HEILBRON_MEYNIEL_2019 or \
       gen_process_type == PROCESS_TYPE_ALTREP_COUPLED:
        gen_process = klass(gen_process_dict["p_change"], p_gen_range)
    elif gen_process_type == PROCESS_TYPE_BERNOULLI_RANDOM_MULTIPLE_CHANGE_POINTS:
        gen_process = klass(gen_process_dict["p_change_list"], p_gen_range)
    elif gen_process_type == PROCESS_TYPE_BERNOULLI_FIXED_CHANGE_POINTS:
        gen_process = klass(gen_process_dict["n_change_points"], p_gen_range)
    else:
        assert False, "unknown type"
    return gen_process
    

def get_p_change(gen_process, n_time_steps):
    if not has_single_p_change(gen_process):
        assert False, "generative process does not have a single volatility level"
    if gen_process.type == PROCESS_TYPE_BERNOULLI_RANDOM_CHANGE_POINTS or \
       gen_process.type == PROCESS_TYPE_MARKOV_COUPLED_RANDOM_CHANGE_POINTS or \
       gen_process.type == PROCESS_TYPE_MARKOV_INDEPENDENT_RANDOM_CHANGE_POINTS or \
       gen_process.type == PROCESS_TYPE_MARKOV_HEILBRON_MEYNIEL_2019 or \
       gen_process.type == PROCESS_TYPE_ALTREP_COUPLED:
        p_change = gen_process.p_change
    elif gen_process.type == PROCESS_TYPE_BERNOULLI_FIXED_CHANGE_POINTS:
        p_change = gen_process.n_change_points / n_time_steps
    else:
        assert False, "unknown type"
    return p_change

def has_single_p_change(gen_process):
    return gen_process.type != PROCESS_TYPE_BERNOULLI_RANDOM_MULTIPLE_CHANGE_POINTS

# Generate multiple sequences data with dimensions as expected by PyTorch models
def generate_netinput_target_probagen(gen_process, n_sequences,
    n_time_steps = SEQ_N_TIME_STEPS):
    p_gen_range = gen_process.p_gen_range
    if gen_process.type == PROCESS_TYPE_BERNOULLI_RANDOM_CHANGE_POINTS:
        p_change = gen_process.p_change
        observation_sequences, p_gen_sequences, _ = generate_sequence_bernoulli_random_change_points(n_sequences, n_time_steps, p_change, p_gen_range)
        netinput, target = get_netinput_target_from_sequences(observation_sequences, p_gen_range)
        probagen = p_gen_sequences
    elif gen_process.type == PROCESS_TYPE_BERNOULLI_RANDOM_MULTIPLE_CHANGE_POINTS:
        p_change_list = gen_process.p_change_list
        assert (n_sequences % len(p_change_list) == 0),\
            """number of sequences should be a multiple of
            the number of volatiliy levels for balance"""
        n_sequences_per_p_change = n_sequences // len(p_change_list)
        netinput_list = []
        target_list = []
        probagen_list = []
        for p_change in p_change_list:
            observation_sequences, p_gen_sequences, _ = generate_sequence_bernoulli_random_change_points(n_sequences_per_p_change, n_time_steps, p_change, p_gen_range)
            netinput_i, target_i = get_netinput_target_from_sequences(observation_sequences, p_gen_range)
            netinput_list.append(netinput_i)
            target_list.append(target_i)
            probagen_list.append(p_gen_sequences)
        netinput = torch.cat(netinput_list, dim=1)
        target = torch.cat(target_list, dim=1)
        probagen = torch.cat(probagen_list, dim=1)
        # shuffle
        randperm_seq_indices = torch.randperm(netinput.size(1))
        netinput = netinput[:, randperm_seq_indices, :]
        target = target[:, randperm_seq_indices, :]
        probagen = probagen[:, randperm_seq_indices]
    elif gen_process.type == PROCESS_TYPE_MARKOV_COUPLED_RANDOM_CHANGE_POINTS or \
         gen_process.type == PROCESS_TYPE_MARKOV_INDEPENDENT_RANDOM_CHANGE_POINTS or \
         gen_process.type == PROCESS_TYPE_MARKOV_HEILBRON_MEYNIEL_2019:
        p_change = gen_process.p_change
        coupled = gen_process.is_markov_coupled()
        with_constraints = gen_process.type == PROCESS_TYPE_MARKOV_HEILBRON_MEYNIEL_2019
        observation_sequences, p_gen_sequences = \
            generate_sequence_markov_random_change_points(
                    n_sequences, n_time_steps, p_change, p_gen_range,
                    coupled=coupled,
                    with_heilbron_meyniel_constraints=with_constraints)
        netinput, target = get_netinput_target_from_sequences(observation_sequences, p_gen_range)
        probagen = p_gen_sequences
    elif gen_process.type == PROCESS_TYPE_ALTREP_COUPLED:
        p_change = gen_process.p_change
        observation_sequences, p_gen_sequences = generate_sequence_altrep(n_sequences, n_time_steps, p_change)
        netinput, target = get_netinput_target_from_sequences(observation_sequences, p_gen_range) # p_gen_range?
        probagen = p_gen_sequences
    elif gen_process.type == PROCESS_TYPE_BERNOULLI_FIXED_CHANGE_POINTS:
        n_change_points = gen_process.n_change_points
        n_time_steps, sequences, probagen = generate_sequences_probagen(n_sequences, n_time_steps, n_change_points, p_gen_range)
        netinput, target = get_netinput_target_from_sequences(sequences, p_gen_range)
    else:
        assert False, "unknown type"
    
    return (netinput, target, probagen)

def get_netinput_target_from_sequences(sequences, p_gen_range):
    # generate one last observation following the sequence
    # to use as the last element of the target vector.
    # for symmetry between all chunks,
    # we assume a change point after the last observation of the sequence,
    # therefore, this observation is sampled uniformly with p in [probagen_min, probagen_max]
    n_time_steps = sequences.shape[0]
    probagen_min, probagen_max = p_gen_range
    p_last = np.random.uniform(low = probagen_min, high = probagen_max)
    target_last = torch.ceil(torch.rand(1) - (1. - p_last)).item()
    # input is the sequence
    netinput = sequences.clone()
    # target is the sequence 1 time step ahead (predict the next observation) plus one last observation
    target = torch.empty_like(sequences)
    target[0 : n_time_steps - 1, :] = sequences[1 : n_time_steps, :]
    target[n_time_steps - 1, :] = target_last
    # PyTorch expects a third dimension
    netinput = netinput.unsqueeze(2)
    target = target.unsqueeze(2)
    return (netinput, target)
    
# Generate functions for fixed change points
def generate_sequences_probagen(n_sequences, n_time_steps_max, n_change_points, p_gen_range):
    # calculate the actual number of time steps
    # which needs to be a multiple of n_chunks
    n_chunks = n_change_points + 1
    n_time_steps_per_chunks = n_time_steps_max // n_chunks
    n_time_steps = n_time_steps_per_chunks * n_chunks
    
    probagen_min, probagen_max = p_gen_range
    probagen_per_chunk = np.random.uniform(low = probagen_min, high = probagen_max, size = (n_chunks, n_sequences))
    probagen = np.repeat(probagen_per_chunk, repeats = n_time_steps_per_chunks, axis = 0)
    probagen = torch.tensor(probagen, dtype = torch.float)
    
    sequences = torch.ceil(torch.rand((n_time_steps, n_sequences)) - (1. - probagen))
    
    return (n_time_steps, sequences, probagen)

# Generate functions for random change points
    
def generate_sequence_bernoulli_random_change_points(n_sequences, n_time_steps, p_change, p_gen_range):
    """Generate sequences of Bernoulli observations in which the probability may change at any time step according to the given probability of change"""
    observation_sequences = torch.empty(n_time_steps, n_sequences, dtype=torch.float)
    p_gen_sequences = torch.empty(n_time_steps, n_sequences, dtype=torch.float)
    change_times_per_sequence = []
    for iSeq in range(n_sequences):
        change_flags = stats.bernoulli.rvs(p_change, size=n_time_steps) # 1 if a change point occurs at time t, 0 if not
        # change point at time t if p_gen[t+1] =/= p_gen[t]
        change_times = np.nonzero(change_flags)
        change_times_per_sequence.append(change_times)
        observation_sequence, p_gen_sequence = generate_sequence_bernoulli_with_change_times(change_times, n_time_steps, lambda: uniform_rvs(p_gen_range))
        observation_sequences[:, iSeq] = observation_sequence
        p_gen_sequences[:, iSeq] = p_gen_sequence
            
    return observation_sequences, p_gen_sequences, change_times_per_sequence

def generate_sequence_markov_random_change_points(n_sequences, n_time_steps, p_change, p_gen_range,
                                                  coupled=True,
                                                  with_heilbron_meyniel_constraints=False):
    """Generate sequences of binary observations according to a Markov process in which the transition probabilities may change all at the same time according to the given probability of change"""
    # p_gen[:, :, 0] = p(x_{t+1} = 1 | x_{t} = 0), p_gen[:, :, 1] = p(x_{t+1} = 1 | x_{t} = 1)
    p_gen_sequences = torch.empty(n_time_steps, n_sequences, 2, dtype=torch.float)
    observation_sequences = torch.empty(n_time_steps, n_sequences, dtype=torch.float)
    for iSeq in range(n_sequences):
        if coupled:
            change_flags = generate_change_flags(p_change, n_time_steps,
                                                 with_max_stable_length_constraint=with_heilbron_meyniel_constraints)
            change_times = get_change_times_from_flags(change_flags)
            observation_sequence, p_gen_sequence = \
                generate_sequence_markov_coupled_with_change_times(
                        change_times, n_time_steps,
                        lambda: torch.tensor([uniform_rvs(p_gen_range), uniform_rvs(p_gen_range)]),
                        with_odd_ratio_constraint=with_heilbron_meyniel_constraints)
        else:
            change_flags = np.zeros((n_time_steps, 2))
            change_flags[:, 0] = generate_change_flags(p_change, n_time_steps,
                        with_max_stable_length_constraint=with_heilbron_meyniel_constraints)
            change_flags[:, 1] = generate_change_flags(p_change, n_time_steps,
                        with_max_stable_length_constraint=with_heilbron_meyniel_constraints)
            observation_sequence, p_gen_sequence = \
                generate_sequence_markov_independent_with_change_flags(
                        change_flags, n_time_steps,
                        lambda: uniform_rvs(p_gen_range),
                        with_odd_ratio_constraint=with_heilbron_meyniel_constraints)
        
        observation_sequences[:, iSeq] = observation_sequence
        p_gen_sequences[:, iSeq, :] = p_gen_sequence
    
    return observation_sequences, p_gen_sequences

def generate_sequence_altrep(n_sequences, n_time_steps, p_change):
    """Generate sequences of binary observations according to a Markov process
    in which the two transition probabilities change from frequent alternations
    to frequent repetitions"""
    p_gen_sequences = torch.empty(n_time_steps, n_sequences, 2, dtype=torch.float)
    observation_sequences = torch.empty(n_time_steps, n_sequences, dtype=torch.float)
    for iSeq in range(n_sequences):
        change_flags = generate_change_flags(p_change, n_time_steps)
        change_times = get_change_times_from_flags(change_flags)
        observation_sequence, p_gen_sequence = \
            generate_sequence_altrep_with_change_times(change_times, n_time_steps)
        
        observation_sequences[:, iSeq] = observation_sequence
        p_gen_sequences[:, iSeq, :] = p_gen_sequence
    
    return observation_sequences, p_gen_sequences

def generate_change_flags(p_change, n_time_steps,
                          with_max_stable_length_constraint=False):
    change_flags = stats.bernoulli.rvs(p_change, size=n_time_steps)
    if with_max_stable_length_constraint:
        while max_stable_length(get_change_times_from_flags(change_flags)) > 300:
            change_flags = stats.bernoulli.rvs(p_change, size=n_time_steps)
    return change_flags
    
def get_change_times_from_flags(change_flags):
    return np.nonzero(change_flags)[0]

def max_stable_length(change_times):
    stable_length = 0
    last_change_time = 0
    for change_time in change_times:
        stable_length = max(stable_length, change_time - last_change_time)
        last_change_time = change_time
    return stable_length

def uniform_rvs(range):
    return range[0] + (range[1] - range[0]) * stats.uniform.rvs()

def generate_sequence_bernoulli_with_change_times(change_times, n_time_steps, p_gen_fun):
    t = 0
    p_gen = p_gen_fun()
    observation_sequence = torch.empty(n_time_steps)
    p_gen_sequence = torch.empty(n_time_steps)
    for t_change in np.append(change_times, n_time_steps - 1):
        observation_sequence[t : t_change + 1] = torch.from_numpy(stats.bernoulli.rvs(p_gen, size=t_change + 1 - t))
        p_gen_sequence[t : t_change + 1] = p_gen
        p_gen = p_gen_fun()
        t = t_change + 1
    return observation_sequence, p_gen_sequence

def generate_sequence_bernoulli_with_change_times_and_pvals(n_time_steps, change_times, p_gen_vals):
    assert len(p_gen_vals) == len(change_times) + 1
    i_p_gen = 0
    t = 0
    observation_sequence = torch.empty(n_time_steps)
    p_gen_sequence = torch.empty(n_time_steps)
    for t_change in np.append(change_times, n_time_steps - 1).astype(int):
        p_gen = p_gen_vals[i_p_gen]
        observation_sequence[t : t_change + 1] = torch.from_numpy(stats.bernoulli.rvs(p_gen, size=t_change + 1 - t))
        p_gen_sequence[t : t_change + 1] = p_gen
        i_p_gen += 1
        t = t_change + 1
    return observation_sequence, p_gen_sequence

def generate_sequence_markov_coupled_with_change_times(change_times, n_time_steps, p_gen_fun, with_odd_ratio_constraint=False):
    t_last_change = 0
    p_gen = p_gen_fun()
    p_gen_sequence = torch.empty(n_time_steps, 2)
    observation_sequence = torch.empty(n_time_steps)
    observation_last = 0
    for t_change in np.append(change_times, n_time_steps - 1):
        # generate observations for current chunk
        p_gen_sequence[t_last_change : t_change + 1, :] = p_gen
        for t in range(t_last_change, t_change + 1):
            if observation_last == 0:
                observation = stats.bernoulli.rvs(p_gen[0])
            elif observation_last == 1:
                observation = stats.bernoulli.rvs(p_gen[1])
            else:
                assert False, "programmer error: observation must be 0 or 1"
            observation_sequence[t] = observation
            observation_last = observation
        # sample probabilities for next chunk (change point)
        p_gen = get_p_gen_val(p_gen, p_gen_fun, with_odd_ratio_constraint=with_odd_ratio_constraint)
        t_last_change = t_change + 1
    return observation_sequence, p_gen_sequence

def generate_sequence_altrep_with_change_times(change_times, n_time_steps, p_gen_epsilon=0.25):
    # when alternation, p_gen_rep is sampled in (0, epsilon)
    # when repetition, p_gen_rep is sampled in (1-epsilon, 1)
    t_last_change = 0
    is_repetition = False
    p_gen_rep = uniform_rvs((1 - p_gen_epsilon, 1)) if is_repetition else uniform_rvs((0, p_gen_epsilon))
    p_gen_sequence = torch.empty(n_time_steps, 2)
    observation_sequence = torch.empty(n_time_steps)
    observation_last = 0
    for t_change in np.append(change_times, n_time_steps - 1):
        # generate observations for current chunk
        p_gen_sequence[t_last_change : t_change + 1, 0] = 1 - p_gen_rep
        p_gen_sequence[t_last_change : t_change + 1, 1] = p_gen_rep
        for t in range(t_last_change, t_change + 1):
            if observation_last == 0:
                observation = stats.bernoulli.rvs(1 - p_gen_rep)
            elif observation_last == 1:
                observation = stats.bernoulli.rvs(p_gen_rep)
            else:
                assert False, "programmer error: observation must be 0 or 1"
            observation_sequence[t] = observation
            observation_last = observation
        # sample probabilities for next chunk (change point)
        is_repetition = not is_repetition
        p_gen_rep = uniform_rvs((1 - p_gen_epsilon, 1)) if is_repetition else uniform_rvs((0, p_gen_epsilon))
        t_last_change = t_change + 1
    return observation_sequence, p_gen_sequence

def generate_sequence_markov_independent_with_change_flags(change_flags, n_time_steps, p_gen_fun, with_odd_ratio_constraint=False):
    p_gen_sequence = torch.empty(n_time_steps, 2)
    observation_sequence = torch.empty(n_time_steps)
    p_gen = torch.empty(2)
    for i in [0, 1]:
        p_gen[i] = p_gen_fun()
    observation_last = 0
    for t in range(n_time_steps):
        p_gen_sequence[t, :] = p_gen
        if observation_last == 0:
            observation = stats.bernoulli.rvs(p_gen[0])
        elif observation_last == 1:
            observation = stats.bernoulli.rvs(p_gen[1])
        else:
            assert False, "programmer error: observation must be 0 or 1"
        observation_sequence[t] = observation
        observation_last = observation
        # sample new generative probability if change point
        for i in [0, 1]:
            if change_flags[t, i] != 0:
                p_gen[i] = get_p_gen_val(p_gen[i], p_gen_fun, with_odd_ratio_constraint=with_odd_ratio_constraint)
    
    return observation_sequence, p_gen_sequence

def get_p_gen_val(p_gen, p_gen_fun, with_odd_ratio_constraint=False):
    new_p_gen = p_gen_fun()
    if with_odd_ratio_constraint:
        while max_odd_ratio_change(p_gen, new_p_gen) < 4.:
            new_p_gen = p_gen_fun()
    return new_p_gen

def max_odd_ratio_change(p_gen, new_p_gen):
    max_value = 1.
    for i in range(len(p_gen)):
        odd_ratio = p_gen[i] / (1. - p_gen[i])
        new_odd_ratio = new_p_gen[i] / (1. - new_p_gen[i])
        odd_ratio_change = max(new_odd_ratio / odd_ratio, odd_ratio / new_odd_ratio)
        max_value = max(max_value, odd_ratio_change)
    return max_value

def generate_sequence_long_short_alternating(n_time_steps, p_c, p_gen_0):
    n_changes = round(p_c * n_time_steps)
    n_changes = n_changes + (1 - n_changes % 2) # want an even number of periods
    n_periods = n_changes + 1
    long_short_ratio = 3
    short_period_length = round(2 * n_time_steps / ((1 + long_short_ratio) * n_periods))
    long_period_length = long_short_ratio * short_period_length
    # compute change times
    t = 0
    is_long_period = 1
    change_times = []
    for i in range(n_changes):
        t += long_period_length if is_long_period == 1 else short_period_length
        change_times.append(t)
        is_long_period = 1 - is_long_period
    # define p_gen function
    global p_gen
    p_gen = p_gen_0
    def p_gen_fun():
        global p_gen
        p_gen = 1 - p_gen
        return p_gen
    observation_sequence, p_gen_sequence = generate_sequence_bernoulli_with_change_times(change_times, n_time_steps, p_gen_fun)
    return observation_sequence, p_gen_sequence, change_times

if __name__ == '__main__': # For debugging
    TEST_SIMPLE = False
    TEST_MULTIPLE = False
    TEST_HEILBRON_MEYNIEL = False
    TEST_MARKOV_INDEPENDENT = False
    TEST_ALTREP_COUPLED = False
    TEST_BERNOULLI_WITH_CHANGETIMES_AND_PVALS = True
    
    if TEST_SIMPLE:
        # Test Bernoulli n sequences
        n_sequences = 4
        n_time_steps = 20
        p_change = 0.25
        p_gen_range = (0.1, 0.9)
    
        observation_sequences, p_gen_sequences, change_times_per_sequence = generate_sequence_bernoulli_random_change_points(n_sequences, n_time_steps, p_change, p_gen_range)
    
        print("observation_sequences size: ", observation_sequences.size())
        print("p_gen_sequences size: ", p_gen_sequences.size())
        print("change_times_per_sequence[1]: ", change_times_per_sequence[1])
        print("observation_sequences[:, 1]", observation_sequences[:, 1])
        print("p_gen_sequences[:, 1]", p_gen_sequences[:, 1])
        
        # Test alternating sequences
        n_time_steps = 384
        p_c = 0.01
        p_gen_0 = .25
        observation_sequence_alt, p_gen_sequence_alt, change_times = generate_sequence_long_short_alternating(n_time_steps, p_c, p_gen_0)
        print("\np_gen_sequence_alt: ", p_gen_sequence_alt)
        
        # Test single Markov sequence
        n_time_steps = 128
        change_times = [9]
        p_gen_fun = lambda: torch.tensor([uniform_rvs(p_gen_range), uniform_rvs(p_gen_range)])
        obs_seq, p_gen_seq = generate_sequence_markov_coupled_with_change_times(change_times, n_time_steps, p_gen_fun)
        print("n_time_steps: ", n_time_steps)
        print("change_times: ", change_times)
        print("p_gen_seq: ", p_gen_seq)
        print("obs_seq: ", obs_seq)
        t_last_change = change_times[-1] + 1
        n_1_1 = sum([1 if obs_seq[t] == 1 and obs_seq[t+1] == 1 else 0 for t in range(t_last_change, n_time_steps - 1)])
        n_1_0 = sum([1 if obs_seq[t] == 0 and obs_seq[t+1] == 1 else 0 for t in range(t_last_change, n_time_steps - 1)])
        n_0_1 = sum([1 if obs_seq[t] == 1 and obs_seq[t+1] == 0 else 0 for t in range(t_last_change, n_time_steps - 1)])
        n_0_0 = sum([1 if obs_seq[t] == 0 and obs_seq[t+1] == 0 else 0 for t in range(t_last_change, n_time_steps - 1)])
        print("stats within [{:d}, {:d}]".format(t_last_change, n_time_steps - 1))
        print("n_1_1: {:d}, n_1_0: {:d}, n_0_1: {:d}, n_0_0: {:d}".format(n_1_1, n_1_0, n_0_1, n_0_0))
        print("f_1_0: {:}", n_1_0 / (n_1_0 + n_0_0))
        print("f_1_1: {:}", n_1_1 / (n_1_1 + n_0_1))
        
        # Test Markov n sequences
        n_sequences = 4
        n_time_steps = 20
        p_change = 0.1
        p_gen_range = (0.1, 0.9)
    
        observation_sequences, p_gen_sequences = generate_sequence_markov_random_change_points(n_sequences, n_time_steps, p_change, p_gen_range)
    
        print("observation_sequences size: ", observation_sequences.size())
        print("p_gen_sequences size: ", p_gen_sequences.size())
        print("observation_sequences[:, 1]", observation_sequences[:, 1])
        print("p_gen_sequences[:, 1]", p_gen_sequences[:, 1])
    
# =============================================================================
    if TEST_MULTIPLE:
        p_change_list = [3 / 384, 5 / 384, 7 / 384]
        gen_process_multiple = GenerativeProcessBernoulliRandomMultiple(p_change_list)
        n_sequences = 80 * 7 * len(p_change_list)
        netinput, target, probagen = generate_netinput_target_probagen(gen_process_multiple, n_sequences)
        print(netinput.size())
        print(target.size())
        print(probagen.size())
        # check that shuffling worked and that
        # actual number of changes is close to the average of p_change_list
        print("theoretical p_change_avg: {:.4f}".format(sum(p_change_list) / len(p_change_list)))
        chunk_size = n_sequences // len(p_change_list)
        for iChunk in range(len(p_change_list)):
            n_changes = 0
            for iSeq in range(iChunk * chunk_size, (iChunk + 1) * chunk_size):
                for t in range(probagen.size(0) - 1):
                    if probagen[t, iSeq] != probagen[t + 1, iSeq]:
                        n_changes += 1
            p_change_avg_actual = n_changes / (chunk_size * probagen.size(0))
            print("actual p_change_avg in {:d}th chunk: {:.4f}".format(iChunk, p_change_avg_actual))

# =============================================================================
    if TEST_HEILBRON_MEYNIEL:
        # Test Heilbron_Meyniel_2019
        gen_process = GenerativeProcessHeilbronMeyniel2019()
        n_sequences = 1
        netinput, target, probagen = generate_netinput_target_probagen(gen_process, n_sequences)
        print(netinput.size())
        print(target.size())
        print(probagen.size())
    
# =============================================================================
    if TEST_MARKOV_INDEPENDENT:
        # Test Markov Independent
        print("Test Markov Independent")
        p_change = 0.1
        gen_process = GenerativeProcessMarkovIndependent(p_change)
        n_sequences = 1
        netinput, target, probagen = generate_netinput_target_probagen(gen_process, n_sequences)
        print("netinput size", netinput.size())
        print("target size", target.size())
        print("probagen", probagen.size())
        change_times_0 = sequence.get_change_times(probagen[:, 0, 0])
        change_times_1 = sequence.get_change_times(probagen[:, 0, 1])
    #    print("p_gen_0", probagen[:, 0, 0])
        print("change_times_0", change_times_0)
        print("change_times_1", change_times_1)

# =============================================================================
    if TEST_ALTREP_COUPLED:
    # Test AltRep Coupled
        print("Test AltRep Coupled")
        p_change = 1/50
        gen_process = GenerativeProcessAltRepCoupled(p_change)
        n_sequences = 1
        inputs, targets, p_gens = generate_netinput_target_probagen(gen_process, n_sequences)
        change_times_0 = sequence.get_change_times(p_gens[:, 0, 0])
        change_times_1 = sequence.get_change_times(p_gens[:, 0, 1])
        print("change_times_0 == change_times_1", change_times_0 == change_times_1)
        print("change_times_0", change_times_0)
        tc0 = change_times_0[0]
        tc1 = change_times_0[1]
        print("p_gens[:tc0+1, 0, 1]", p_gens[:tc0+1, 0, 1])
        print("1 - p_gens[:tc0+1, 0, 0]", 1 - p_gens[:tc0+1, 0, 0])
        print("p_gens[tc0+1:tc1+1, 0, 1]", p_gens[tc0+1:tc1+1, 0, 1])
        print("1 - p_gens[tc0+1:tc1+1, 0, 0]", 1 - p_gens[tc0+1:tc1+1, 0, 0])
        print("inputs[:tc0+1, 0, 0]", inputs[:tc0+1, 0, 0])
        print("inputs[tc0+1:tc1+1, 0, 0]", inputs[tc0+1:tc1+1, 0, 0])

    if TEST_BERNOULLI_WITH_CHANGETIMES_AND_PVALS:
        observations, p_gens = generate_sequence_bernoulli_with_change_times_and_pvals(20, [], [0.5])
        print("observations, p_gens", observations, p_gens)
        observations, p_gens = generate_sequence_bernoulli_with_change_times_and_pvals(20, [15], [0.2, 1.0])
        print("observations, p_gens", observations, p_gens)
