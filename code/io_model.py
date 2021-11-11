import generate as gen
import numpy as np
import sequence
import torch
import TransitionProbModel.MarkovModel_Python.IdealObserver as IO

class IOModel(object):
    def __init__(self, gen_process):
        super(IOModel, self).__init__()
        self.gen_process = gen_process

    @classmethod
    def get_class_label(cls):
        return "Optimal"
    
    def get_label(self, detailed=False, with_coupled_label=False):
        label = self.get_class_label()
        if with_coupled_label:
            label += " (Independent)" if self.gen_process.is_markov_independent() \
                else " (Coupled)"
        return label
    
    def __call__(self, inputs):
        return self.get_predictions(inputs)
    
    def get_predictions(self, inputs):
        output_means = self.get_output_stats(inputs)["mean"]
        if self.gen_process.is_markov():
            return sequence.get_relevant_estimates_given_inputs(output_means, inputs)
        else:
            return output_means
    
    def get_output_stats(self, inputs, resol=20):
        return self.get_outputs(inputs, include_dist=False, resol=resol)

    def get_outputs(self, inputs, include_dist=True, resol=20):
        gen_process = self.gen_process
        # create tensor for outputs
        n_estimates = 2 if gen_process.is_markov() else 1 # 2 estimates for 2 transition probabilities
        n_time_steps = inputs.size(0)
        n_sequences = inputs.size(1)
        size = (n_time_steps, n_sequences, n_estimates)
        output_means = torch.empty(size, dtype=inputs.dtype, layout=inputs.layout, device=inputs.device)
        output_sds = torch.empty(size, dtype=inputs.dtype, layout=inputs.layout, device=inputs.device)
        if include_dist:
            size_dist = (n_time_steps, n_sequences, n_estimates, resol)
            output_dists = torch.empty(size_dist, dtype=inputs.dtype, layout=inputs.layout, device=inputs.device)
        # get IdealObserver arguments
        order = 1 if gen_process.is_markov() else 0
        p_change = gen.get_p_change(gen_process, n_time_steps)
        if gen_process.p_change == 0:
            obs_type = "fixed"
        elif gen_process.is_markov_independent():
            obs_type = "hmm_uncoupled"
        else:
            obs_type = "hmm"
        options = {
                "p_c": p_change,
                "resol": resol
                }
        # fill in outputs by computing IO estimates for each sequence
        for iSeq in range(n_sequences):
            sequence = inputs[:, iSeq, 0].detach().numpy()
            io_out = IO.IdealObserver(sequence, obs_type, order=order, options=options)
            if order == 1:
                io_p01_mean = io_out[(0, 1)]["mean"] # 1 | 0
                io_p11_mean = io_out[(1, 1)]["mean"] # 1 | 1
                io_p01_sd = io_out[(0, 1)]["SD"]
                io_p11_sd = io_out[(1, 1)]["SD"]
                output_means[:, iSeq, 0] = torch.tensor(io_p01_mean)
                output_means[:, iSeq, 1] = torch.tensor(io_p11_mean)
                output_sds[:, iSeq, 0] = torch.tensor(io_p01_sd)
                output_sds[:, iSeq, 1] = torch.tensor(io_p11_sd)
                if include_dist:
                    io_p01_dist = io_out[(0, 1)]['dist']
                    io_p11_dist = io_out[(1, 1)]['dist']
                    io_p01_dist = np.swapaxes(io_p01_dist, 0, 1)
                    io_p11_dist = np.swapaxes(io_p11_dist, 0, 1)
                    output_dists[:, iSeq, 0, :] = torch.tensor(io_p01_dist)
                    output_dists[:, iSeq, 1, :] = torch.tensor(io_p11_dist)
            elif order == 0:
                io_p0_mean = io_out[(0,)]["mean"]
                io_p0_sd = io_out[(0,)]["SD"]
                output_means[:, iSeq, 0] = torch.tensor(1 - io_p0_mean)
                output_sds[:, iSeq, 0] = torch.tensor(io_p0_sd)
                if include_dist:
                    io_p0_dist = io_out[(0,)]['dist']
                    io_p0_dist = np.swapaxes(io_p0_dist, 0, 1)
                    output_dists[:, iSeq, 0, :] = torch.tensor(io_p0_dist)
        
        outputs = dict(mean=output_means, sd=output_sds)
        if include_dist:
            outputs["dist"] = output_dists
        return outputs
        


if __name__ == '__main__':
    n_time_steps = 100
    n_sequences = 2
    
    #%% Bernoulli, no change points
    gen_process = gen.GenerativeProcessBernoulliRandom(0)
    io_model = IOModel(gen_process)
    inputs, _, p_gens = gen.generate_netinput_target_probagen(gen_process, n_sequences, n_time_steps_max = n_time_steps)
    io_out = io_model(inputs)
    print("inputs: ", inputs[:, 0, 0].detach().numpy())
    print("p_gens: ", p_gens[:, 0].detach().numpy())
    print("io_out: ", io_out[:, 0, 0].detach().numpy())
    #%% Bernoulli, with change points
    gen_process = gen.GenerativeProcessBernoulliRandom(4 / n_time_steps)
    io_model = IOModel(gen_process)
    inputs, _, p_gens = gen.generate_netinput_target_probagen(gen_process, n_sequences, n_time_steps_max = n_time_steps)
    io_out = io_model(inputs)
    print("inputs: ", inputs[:, 0, 0].detach().numpy())
    print("p_gens: ", p_gens[:, 0].detach().numpy())
    print("io_out: ", io_out[:, 0, 0].detach().numpy())
    #%% Markov, with change points
    gen_process = gen.GenerativeProcessMarkovCoupled(1 / n_time_steps)
    io_model = IOModel(gen_process)
    inputs, _, p_gens = gen.generate_netinput_target_probagen(gen_process, n_sequences, n_time_steps_max = n_time_steps)
    io_out = io_model(inputs)
    print("p_gens: ", p_gens[:, 0].detach().numpy())
    print("io_out: ", io_out[:, 0].detach().numpy())
