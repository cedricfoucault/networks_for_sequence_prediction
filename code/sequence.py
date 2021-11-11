# functions for processing sequence data

def get_numpy1D_from_tensor(tensor):
    return tensor.flatten().detach().numpy()

def sample_from_tensor(tensor, i_seq):
    tensor_seq_i = tensor[:, i_seq, 0] if len(tensor.size()) == 3 else tensor[:, i_seq]
    return tensor_seq_i.reshape(-1).detach().numpy()

def get_change_times(p_gen):
    return [t for t in range(len(p_gen) - 1) if p_gen[t + 1] != p_gen[t]]

def get_relevant_estimates_given_inputs(estimates, inputs):
    assert estimates.size(2) == 2, "this only applies to estimates conditional on binary input"
    # estimates[:, :, 0] when inputs == 0, estimates[:, :, 1] when x == 1
    return estimates[:, :, 0].unsqueeze(2) * (1 - inputs) + \
           estimates[:, :, 1].unsqueeze(2) * inputs


