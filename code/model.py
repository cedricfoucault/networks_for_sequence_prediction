import collections
import copy
import hyperparam_config as hconfig
import numpy as np
import numpy.polynomial
import torch

def get_class_from_string(string):
    if string.lower() == "deltarule":
        return DeltaRule
    elif string.lower() == "leakyintegrator":
        return LeakyIntegrator
    else:
        return Network

def get_new_model_instance_with_config(config):
    model_class = config.get(hconfig.MODEL_CLASS_KEY, "network")
    model_instance = get_class_from_string(model_class).new_with_config(config)
    return model_instance

# Define the network
class Network(torch.nn.Module):
    @classmethod
    def load(cls, path):
        saved_data = torch.load(path)
        if "config" in saved_data:
            net = cls.new_with_config(saved_data["config"])
            statedict = saved_data["state_dict"]
            net.load_state_dict(statedict)
        else:
            assert False, "model could not be loaded from given file"
        return net
    
    @classmethod
    def new_with_config(cls, config):
        net = cls()
        net.init_with_config(config)
        return net
    
    def __init__(self):
        super(Network, self).__init__()
    
    def init_with_config(self, config):
        self.config = config
        self.init_layers_with_config(config)
        self.init_parameters()
    
    def init_layers_with_config(self, config):
        unit_type = config[hconfig.UNIT_TYPE_KEY]
        n_units = config[hconfig.N_UNITS_KEY]
        n_layers = config[hconfig.N_LAYERS_KEY] if hconfig.N_LAYERS_KEY in config else 1
        hidden_layer_fun = getattr(torch.nn, unit_type)
        hidden_layer_kwargs = {
                "input_size": 1,
                "hidden_size": n_units,
                "num_layers": n_layers,
                "bias": True
                }
        if unit_type.upper() == "RNN":
            hidden_layer_kwargs["nonlinearity"] = "tanh"
        self.input_bias = torch.nn.Parameter(torch.tensor(0.))
        self.hidden_layer = hidden_layer_fun(**hidden_layer_kwargs)
        self.output_layer = torch.nn.Linear(self.hidden_layer.hidden_size, 1, bias = True)
        self.output_activation = getattr(torch, "sigmoid")
    
    def init_parameters(self):
        n_hunits = self.hidden_layer.hidden_size
        n_gates = self.get_n_gates()
        n_layers = self.hidden_layer.num_layers
        has_diagonal_init = self.has_diagonal_init()
        # if unspecified, the standard deviation of the distribution
        # from which a given set of weights are drawn defaults to:
        # std = 1 / sqrt(N)
        # following Xavier Glorot & Bengio's scheme
        # since fan-in, the number of input connection a unit receives
        # is equal to N (output unit) or N+1 (recurrent units).
        std_w_ih_default = 1. / (n_hunits ** (1/2))
        std_w_hh_default = 1. / (n_hunits ** (1/2))
        std_w_ho_default = 1. / (n_hunits ** (1/2))
        std_w_ih = self.config.get(hconfig.INITIALIZATION_WEIGHT_INPUT_TO_HIDDEN_STD_KEY, std_w_ih_default)
        std_w_hh = self.config.get(hconfig.INITIALIZATION_WEIGHT_HIDDEN_TO_HIDDEN_STD_KEY, std_w_hh_default)
        std_w_ho = self.config.get(hconfig.INITIALIZATION_WEIGHT_HIDDEN_TO_OUTPUT_STD_KEY, std_w_ho_default)
        assert n_layers <= 3, "no more than 3 layers supported"
        ih_tensors = [self.hidden_layer.weight_ih_l0]
        hh_tensors = [self.hidden_layer.weight_hh_l0]
        if n_layers >= 2:
            ih_tensors.append(self.hidden_layer.weight_ih_l1)
            hh_tensors.append(self.hidden_layer.weight_hh_l1)
        if n_layers >= 3:
            ih_tensors.append(self.hidden_layer.weight_ih_l2)
            hh_tensors.append(self.hidden_layer.weight_hh_l2)
        # input-to-hidden weights
        for ih_tensor in ih_tensors:
            torch.nn.init.normal_(ih_tensor.data, std=std_w_ih)
        # hidden-to-hidden (recurrent) weights
        for hh_tensor in hh_tensors:
            if has_diagonal_init:
                # set off-diagonal weights to 0 if diagonal initialization
                diag_coefs = torch.randn(n_gates + 1, n_hunits) * std_w_hh
                hh_tensor.data = torch.diag_embed(diag_coefs).reshape((n_gates + 1) * n_hunits, n_hunits)
            else: 
                torch.nn.init.normal_(hh_tensor.data, std=std_w_hh)
            init_diagonal_mean = self._init_diagonal_mean()
            if init_diagonal_mean != 0:
                init_diagonal_mean_coefs = torch.ones(n_gates + 1, n_hunits) * init_diagonal_mean
                init_diagonal_mean_tensor = torch.diag_embed(init_diagonal_mean_coefs).reshape((n_gates + 1) * n_hunits, n_hunits)
                hh_tensor.data = hh_tensor.data + init_diagonal_mean_tensor
        # hidden-to-output weights
        torch.nn.init.normal_(self.output_layer.weight.data, std=std_w_ho)

    def has_diagonal_init(self):
        return self.config.get(hconfig.INITIALIZATION_SCHEME_KEY, "default").lower() == "diagonal"

    def has_train_diagonal_only(self):
        return self.config.get(hconfig.TRAIN_DIAGONAL_ONLY_KEY, False)

    def has_diagonal_recurrent_weight_matrix(self):
        return self.has_diagonal_init() and self.has_train_diagonal_only()

    def _init_diagonal_mean(self):
        return self.config.get(hconfig.INIT_DIAGONAL_MEAN_KEY, 0)

    def has_train_outputonly(self):
        return self.config.get(hconfig.TRAIN_OUTPUT_ONLY_KEY, False)

    def _zero_nondiagonal_weights_gradient(self):
        n_hunits = self.hidden_layer.hidden_size
        n_gates = self.get_n_gates()
        n_layers = self.hidden_layer.num_layers
        hh_tensors = [self.hidden_layer.weight_hh_l0]
        if n_layers >= 2:
            hh_tensors.append(self.hidden_layer.weight_hh_l1)
        if n_layers >= 3:
            hh_tensors.append(self.hidden_layer.weight_hh_l2)
        for hh_tensor in hh_tensors:
            mask_coefs = torch.ones(n_gates + 1, n_hunits)
            hh_tensor.grad.data = hh_tensor.grad.data * torch.diag_embed(mask_coefs).reshape((n_gates + 1) * n_hunits, n_hunits)

    def save(self, path):
        if hasattr(self, "config"):
            saved_data = {
                    "state_dict": self.state_dict(),
                    "config": self.config
                    }
        else:
            assert False, "could not save: missing config"
            saved_data = { "state_dict": self.state_dict() }
        torch.save(saved_data, path)

    def extract_parameters_dict(self):
        return copy.deepcopy(self.state_dict())

    def restore_parameters_dict(self, parameters_dict):
        self.load_state_dict(parameters_dict)

    def forward(self, x):
        logits = self.forward_logits(x)
        return self.output_activation(logits)

    def forward_logits(self, x):
        i = x + self.input_bias
        h, _ = self.hidden_layer(i)
        return self.output_layer(h)
        
    def get_gradient_norm(self):
        norm_squared = 0
        for p in self.parameters():
            param_norm_squared = p.grad.data.norm(2) ** 2
            norm_squared += param_norm_squared.item()
        return norm_squared ** (1. / 2)

    def has_GRU(self):
        return isinstance(self.hidden_layer, torch.nn.GRU)

    def has_LSTM(self):
        return isinstance(self.hidden_layer, torch.nn.LSTM)

    def get_n_gates(self):
        if self.has_GRU():
            return 2
        elif self.has_LSTM():
            return 3
        else:
            return 0

    def forward_activations(self, inputs, sanity_check=True):
        assert self.hidden_layer.num_layers <= 1, "not implemented for more than 1 layer"
        assert not self.has_LSTM(), "not implemented for LSTM"
        has_GRU = self.has_GRU()
        # create tensors to store activations
        hidden = torch.empty(inputs.size(0), inputs.size(1), self.hidden_layer.hidden_size)
        if has_GRU:
            reset_gate = torch.empty(inputs.size(0), inputs.size(1), self.hidden_layer.hidden_size)
            new_state = torch.empty(inputs.size(0), inputs.size(1), self.hidden_layer.hidden_size)
            update_gate = torch.empty(inputs.size(0), inputs.size(1), self.hidden_layer.hidden_size)
        # define forward hook to compute activations
        def _activations_hook(module, input, output):
            hidden.data = output[0].data
            if has_GRU:
                ## Retreive weight and bias parameters
                # r: parameters for reset gate
                # z: parameters for update gate
                # n: parameters for "new" hidden state (equivalent to Elman network)
                # input-hidden weights concatenated (W_ir|W_iz|W_in),
                # shape (3*hidden_size, input_size)
                w_ih = module.weight_ih_l0
                # hidden-hidden weights concatenated (W_hr|W_hz|W_hn),
                # shape (3*hidden_size, hidden_size)
                w_hh = module.weight_hh_l0
                # input-hidden bias concatenated (b_ir|b_iz|b_in),
                # shape (3*hidden_size)
                b_ih = module.bias_ih_l0
                # hidden-hidden bias concatenated (b_hr|b_hz|b_hn),
                # shape (3*hidden_size)
                b_hh = module.bias_hh_l0
                ## Retrieve input and hidden state
                input, = input
                hidden_previous = torch.roll(hidden, +1, 0)
                hidden_previous[0, :, :] = 0.
                ## Compute gate and new  activations
                ## from input and previous hidden state
                gi = torch.matmul(input, w_ih.t()) + b_ih
                gh = torch.matmul(hidden_previous, w_hh.t()) + b_hh
                i_r, i_z, i_n = gi.chunk(3, dim=2)
                h_r, h_z, h_n = gh.chunk(3, dim=2)
                reset_gate.data = torch.sigmoid(i_r + h_r)
                update_gate.data = torch.sigmoid(i_z + h_z)
                new_state.data = torch.tanh(i_n + reset_gate * h_n)
                ## Sanity check:
                ## if computations are correct
                ## hidden state computed from gate and new state
                ## should be equal to the retrieved hidden state
                if sanity_check:
                    hy = (1 - update_gate) * new_state + update_gate * hidden_previous
                    assert hidden.allclose(hy, atol=1e-3), \
                        "calculation error or imprecision"
                # In a multilayer GRU, the input "x^t" of the l-th layer (l>=2)
                # is the hidden state h^t of the (l-1)-th layer
            
        # perform forward computations on given inputs
        hook = self.hidden_layer.register_forward_hook(_activations_hook)
        outputs = self(inputs)
        hook.remove()
        # return activations as a single object
        activations = dict(
                hidden_state=hidden,
                outputs=outputs
                )
        if has_GRU:
            activations["reset_gate"] = reset_gate
            activations["new_state"] = new_state
            activations["update_gate"] = update_gate
        return activations
    
    def compute_next_activations(self, inputs_t, hidden_state_tminus1):
        assert self.hidden_layer.num_layers <= 1, "not implemented for more than 1 layer"
        assert (len(inputs_t.size()) == 2) and (len(hidden_state_tminus1.size()) == 2),\
            "this should be given only inputs at time t and hidden state at time t-1"
        assert not self.has_LSTM(), "not implemented for LSTM"
        has_GRU = self.has_GRU()
        input_hidden_t = inputs_t + self.input_bias # /!\ watch out for self.input_bias /!\
        w_ih = self.hidden_layer.weight_ih_l0
        b_ih = self.hidden_layer.bias_ih_l0
        w_hh = self.hidden_layer.weight_hh_l0
        b_hh = self.hidden_layer.bias_hh_l0
        if has_GRU:
            ## Compute gate and new  activations
            ## from input and previous hidden state
            gi = torch.matmul(input_hidden_t, w_ih.t()) + b_ih
            gh = torch.matmul(hidden_state_tminus1, w_hh.t()) + b_hh
            i_r, i_z, i_n = gi.chunk(3, dim=1)
            h_r, h_z, h_n = gh.chunk(3, dim=1)
            reset_gate_t = torch.sigmoid(i_r + h_r)
            update_gate_t = torch.sigmoid(i_z + h_z)
            new_state_t = torch.tanh(i_n + reset_gate_t * h_n)
            hidden_state_t = (1 - update_gate_t) * new_state_t + update_gate_t * hidden_state_tminus1
        else:
            x_ih = torch.matmul(input_hidden_t, w_ih.t()) + b_ih
            x_hh = torch.matmul(hidden_state_tminus1, w_hh.t()) + b_hh
            hidden_state_t = torch.tanh(x_ih + x_hh)
        outputs_t = self.compute_outputs_from_hidden_state(hidden_state_t)
        activations = dict(
                hidden_state=hidden_state_t,
                outputs=outputs_t
                )
        if has_GRU:
            activations["reset_gate"] = reset_gate_t
            activations["new_state"] = new_state_t
            activations["update_gate"] = update_gate_t
        return activations
    
    def compute_outputs_from_hidden_state(self, hidden_state):
        o = self.output_layer(hidden_state)
        outputs = self.output_activation(o)
        return outputs
    
    def get_label(self, detailed=False):
        label = "Network"
        if detailed:
            architecture_label = self.get_architecture_type_label()
            n_layers_label = self.get_n_layers_label()
            n_units_label = self.get_n_units_label()
            detail_label_components = [ label \
                for label in [architecture_label, n_layers_label, n_units_label] \
                if label]
            detail_label = ", ".join(detail_label_components)
            label += f" ({detail_label})"
        return label

    def get_architecture_label(self, detailed=False):
        label = self.get_architecture_type_label()
        if detailed:
            n_layers_label = self.get_n_layers_label()
            n_units_label = self.get_n_units_label()
            detail_label_components = [ label \
                for label in [n_layers_label, n_units_label] \
                if label]
            detail_label = ", ".join(detail_label_components)
            label += f" ({detail_label})"

        return label

    def get_mechanisms_label(self, detailed=False, include_gating_type=False):
        unit_type = self.get_unit_type_label()
        if self.has_diagonal_recurrent_weight_matrix():
            return "Without lateral\nconnections"
        elif unit_type.lower() == "elman":
            return "Without gating"
        elif self.has_train_outputonly():
            return "Without recurrent\nweight training"
        else:
            label = "Gated recurrent\nnetwork"
            if include_gating_type:
                label += f" ({unit_type})"
            return label


    def get_architecture_type_label(self):
        unit_type = self.get_unit_type_label()
        if self.has_diagonal_recurrent_weight_matrix():
            return "Diagonal " + unit_type
        else:
            return unit_type

    def get_unit_type_label(self):
        unit_type = self.config[hconfig.UNIT_TYPE_KEY]
        if unit_type == "RNN":
            unit_type = "Elman"
        return unit_type

    def get_n_layers_label(self):
        n_layers = self.hidden_layer.num_layers
        if n_layers > 1:
            return f"{n_layers:d} layers"
        else:
            return None

    def get_n_units_label(self):
        n_units = self.config[hconfig.N_UNITS_KEY]
        return f"{n_units:d} units"

    def get_n_trainable_parameters(self, check_analytical_formula=True):
        # Compute number of parameters using PyTorch enumeration on the layers which were trained
        if self.has_train_outputonly():
            trainable_parameters = self.output_layer.parameters()
        else:
            trainable_parameters = self.parameters()
        count = 0
        for param in trainable_parameters:
            count += param.numel()
        # Subtract coefficients of the recurrent weight matrix which were not trained in the diagonal case
        unit_type = self.get_unit_type_label()
        n_hunits = self.hidden_layer.hidden_size
        has_diagonal_recurrent_weight_matrix = self.has_diagonal_recurrent_weight_matrix()
        if self.has_diagonal_recurrent_weight_matrix():
            if unit_type.lower() == "gru":
                count -= 3 * n_hunits * (n_hunits - 1)
            elif unit_type.lower() == "elman":
                count -= n_hunits * (n_hunits - 1)
            else:
                assert False, "not implemented"
        # Below are analytical calculations that should lead to the same number.
        # I added a check to see if the number of parameters gotten from pytorch
        # is equal to the number of parameters I derived on paper.
        # The check is there to prevent potential regressions:
        # in case the two diverge later on, I should inspect what's going on
        if check_analytical_formula:
            count_analytical = self._get_n_trainable_parameters_at_n_hunits_analytically(n_hunits)
            assert count_analytical == count,\
                f"Mismatch between calculated vs. actual num. of trainable parameters: {count_analytical} vs. {count}"

        return count

    def _get_n_trainable_parameters_at_n_hunits_analytically(self, n_hunits):
        poly_coefficients = self._get_n_trainable_parameters_polynomial_coefficients_of_n_hunits()
        n_params = int(np.polynomial.polynomial.polyval(n_hunits, poly_coefficients))
        return n_params

    def _get_n_trainable_parameters_polynomial_coefficients_of_n_hunits(self):
        """
        returns the coefficients of the polynomial that gives
        the number of trainable parameters
        as a function of the number of hidden units
        """
        unit_type = self.get_unit_type_label()
        has_diagonal_recurrent_weight_matrix = self.has_diagonal_recurrent_weight_matrix()
        train_outputonly = self.has_train_outputonly()
        if train_outputonly:
            # 1 + n_hunits
            poly_coefficients = (1, 1)
        else:
            if unit_type.lower() == "gru":
                if has_diagonal_recurrent_weight_matrix:
                    # 2 + 13 * n_hunits
                    poly_coefficients = (2, 13)
                else:
                    # 2 + 10 * n_hunits + 3 * n_hunits ** 2 
                    poly_coefficients = (2, 10, 3)
            elif unit_type.lower() == "elman":
                if has_diagonal_recurrent_weight_matrix:
                    # 2 + 5 * n_hunits
                    poly_coefficients = (2, 5)
                else:
                    # 2 + 4 * n_hunits + 1 * n_hunits ** 2
                    poly_coefficients = (2, 4, 1)
            else:
                assert False, "not implemented for this unit type"
                poly_coefficients = None

        return poly_coefficients
        

class DeltaRule(torch.nn.Module):
    @classmethod
    def new_with_config(cls, config):
        dr = cls(config)
        return dr
    
    def __init__(self, config=None):
        super(DeltaRule, self).__init__()
        self.lr = torch.nn.Parameter(torch.tensor(0.))
        self.config = config if (config is not None) else {}
        self.init_parameters()
    
    def init_parameters(self):
        # log uniform lr initialization in 1e-3, 1e0
        loglr = torch.tensor(0.)
        torch.nn.init.uniform_(loglr, a=-2.5, b=-0.5)
        self.lr.data = 10 ** loglr
        
    @classmethod
    def load(cls, path):
        saved_data = torch.load(path)
        dr = cls.new_with_config(saved_data["config"])
        statedict = saved_data["state_dict"]
        dr.load_state_dict(statedict)
        return dr
        
    def save(self, path):
        saved_data = {
                "state_dict": self.state_dict(),
                "config": self.config
                }
        torch.save(saved_data, path)
        
    def forward(self, x):
        return self.get_predictions(x)
    
    def get_predictions(self, inputs):
        p_estimates = self.get_p_estimates(inputs)
        if self.get_order() == 1:
            # p(1 | 0) when x == 0, p(1 | 1) when x == 1
            return p_estimates[:, :, 0].unsqueeze(2) * (1 - inputs) + \
                   p_estimates[:, :, 1].unsqueeze(2) * inputs
        else:
            return p_estimates
    
    
    def get_p_estimates(self, inputs):
        n_time_steps = inputs.size(0)
        n_seqs = inputs.size(1)
        order = self.get_order()
        p = inputs.new_zeros((n_time_steps, n_seqs, order + 1))
        p_prev = inputs.new_ones((n_seqs, order + 1)) * 0.5 # prior
        obs_prev = inputs.new_zeros((n_seqs))
        for t in range(n_time_steps):
            effective_lr = self.lr.clamp(0., 1.)
            if order == 1:
                p_prev[:, 0] = p_prev[:, 0] + effective_lr * (inputs[t, :, 0] - p_prev[:, 0]) * (1 - obs_prev[:])
                p_prev[:, 1] = p_prev[:, 1] + effective_lr * (inputs[t, :, 0] - p_prev[:, 1]) * obs_prev[:]
            else:
                p_prev = p_prev + effective_lr * (inputs[t, :, :] - p_prev)
            
            p[t, :, :] = p_prev
            obs_prev = inputs[t, :, 0]
        return p
        
    def get_gradient_norm(self):
        return self.lr.grad.norm()
    
    def get_order(self):
        return 1 if self.config.get(hconfig.ESTIMATE_TYPE, "item").lower() == "transition" else 0
    
    def get_label(self, detailed=False):
        label = "Delta-rule"
        if detailed:
            order = self.get_order()
            if order == 1:
                label += " (bigrams)"
            else:
                label += " (unigram)"
        return label
    
    
class LeakyIntegrator(torch.nn.Module):
    @classmethod
    def new_with_config(cls, config):
        instance = cls(config)
        return instance
    
    def __init__(self, config=None):
        super(LeakyIntegrator, self).__init__()
        # the mean lifetime of the exponential decay,
        # a.k.a omega in Maheu & Meyniel (2016)
        # self.lifetime = torch.nn.Parameter(torch.tensor(1.))
        # decay_rate == exp(- 1 / omega)
        self.decay_rate = torch.nn.Parameter(torch.tensor(1.))
        self.config = config if (config is not None) else {}
        self.init_parameters()
    
    def init_parameters(self):
        # log-uniform initialization of lifetime in [1, 1000]
        loglifetime = torch.tensor(0.)
        torch.nn.init.uniform_(loglifetime, a=0.5, b=2.5)
        lifetime = 10 ** loglifetime
        self.decay_rate.data = torch.exp(- 1 / lifetime)
        # self.lifetime.data = 10 ** loglft
        
    @classmethod
    def load(cls, path):
        saved_data = torch.load(path)
        instance = cls.new_with_config(saved_data["config"])
        statedict = saved_data["state_dict"]
        instance.load_state_dict(statedict)
        return instance
        
    def save(self, path):
        saved_data = {
                "state_dict": self.state_dict(),
                "config": self.config
                }
        torch.save(saved_data, path)
        
    def forward(self, x):
        return self.get_predictions(x)
    
    def get_predictions(self, inputs):
        p_estimates = self.get_p_estimates(inputs)
        if self.get_order() == 1:
            # p(1 | 0) when x == 0, p(1 | 1) when x == 1
            return p_estimates[:, :, 0].unsqueeze(2) * (1 - inputs) + \
                   p_estimates[:, :, 1].unsqueeze(2) * inputs
        else:
            return p_estimates
    
    def get_p_estimates(self, inputs):
        n_time_steps = inputs.size(0)
        n_seqs = inputs.size(1)
        order = self.get_order()
        n_ones_leaky = inputs.new_zeros((n_time_steps, n_seqs, order + 1))
        n_all_leaky = inputs.new_zeros((n_time_steps, n_seqs, order + 1))
        if order == 1:
            # we calculate the leaky count of
            # - 1|0: one observations, conditioned on 0
            # - 1:1: one observations, conditioned on 1
            # - 1+0|0: all observations, conditioned on 0
            # - 1+0|1: all observations, conditioned on 1
            n_1_0_leaky_t = inputs.new_zeros((n_seqs))
            n_1_1_leaky_t = inputs.new_zeros((n_seqs))
            n_all_0_leaky_t = inputs.new_zeros((n_seqs))
            n_all_1_leaky_t = inputs.new_zeros((n_seqs))
            input_prev = 0.
            for t in range(n_time_steps):
                n_1_0_leaky_t = self.decay_rate * (
                    (n_1_0_leaky_t + inputs[t, :, 0]) * (1 - input_prev) + \
                    n_1_0_leaky_t * input_prev 
                )
                n_all_0_leaky_t = self.decay_rate * (
                    (n_all_0_leaky_t + 1) * (1 - input_prev) + \
                    n_all_0_leaky_t * input_prev
                )
                n_1_1_leaky_t = self.decay_rate * (
                    (n_1_1_leaky_t + inputs[t, :, 0]) * input_prev + \
                    n_1_1_leaky_t * (1 - input_prev)
                )
                n_all_1_leaky_t = self.decay_rate * (
                    (n_all_1_leaky_t + 1) * input_prev + \
                    n_all_1_leaky_t * (1 - input_prev)
                )
                
                n_ones_leaky[t, :, 0] = n_1_0_leaky_t
                n_ones_leaky[t, :, 1] = n_1_1_leaky_t
                n_all_leaky[t, :, 0] = n_all_0_leaky_t
                n_all_leaky[t, :, 1] = n_all_1_leaky_t
                input_prev = inputs[t, :, 0]
        else: 
            # we calculate the leaky count of one observations
            # and the leaky count of all observations
            n_ones_leaky = inputs.new_zeros((n_time_steps, n_seqs, order + 1))
            n_all_leaky = inputs.new_zeros((n_time_steps, n_seqs, order + 1))
            n_ones_leaky_t = inputs.new_zeros((n_seqs))
            n_all_leaky_t = inputs.new_zeros((n_seqs))
            for t in range(n_time_steps):
                n_ones_leaky_t = self.decay_rate * n_ones_leaky_t + inputs[t, :, 0]
                n_all_leaky_t = self.decay_rate * n_all_leaky_t + 1
                n_ones_leaky[t, :, 0] = n_ones_leaky_t
                n_all_leaky[t, :, 0] = n_all_leaky_t
        ## endif
        # prior on the counts
        n_ones_prior = 1
        n_all_prior = 2
        # p(1) is the ratio of the leaky count of ones
        # to the leaky count of all observations
        p = (n_ones_leaky + n_ones_prior) / (n_all_leaky + n_all_prior)
        return p
        
    def get_gradient_norm(self):
        # return self.lifetime.grad.norm()
        return self.decay_rate.grad.norm()
    
    def get_order(self):
        return 1 if self.config.get(hconfig.ESTIMATE_TYPE, "item").lower() == "transition" else 0
    
    def get_label(self, detailed=False):
        label = "Leaky"
        if detailed:
            order = self.get_order()
            if order == 1:
                label += " (bigrams)"
            else:
                label += " (unigram)"
        return label
    
    def get_lifetime(self):
        return - 1 / torch.log(self.decay_rate)

##### DEBUGGING #####

if __name__ == '__main__':
    import data
    import measure
    import train
    
    TEST_NON_NETWORK = False
    TEST_NETWORK_ACTIVATIONS = True
    
    if TEST_NON_NETWORK:
        for dataset_name in ["B_PC1by75_NMB160_28-02-20", "MI_PC1by75_NMB400_01-03-20", "MC_PC1by75_NMB400_28-02-20"]:
            train_data, test_data, gen_process = data.load_train_test_data(dataset_name)
            n_minibatches = 30
            train_data = train_data[0][0:n_minibatches], train_data[1][0:n_minibatches], train_data[2][0:n_minibatches]
            test_inputs, test_targets, test_probagen = test_data
            chance_loss = measure.get_chance_loss(test_inputs, test_targets)
            print("dataset: ", dataset_name)
            print("chance loss: {:.4f}".format(chance_loss))
            models = []
            for model_class in [DeltaRule, LeakyIntegrator]:
                # Train
                config = { "optimizer.name": "SGD", "optimizer.lr": 0.003 }
                if gen_process.is_markov():
                    config[hconfig.ESTIMATE_TYPE] = "transition"
                model = model_class(config=config)
                optimizer = train.get_optimizer_with_config(model.parameters(), config)
                train.train_network(train_data, model, optimizer,
                                    progress_callback=lambda model, iMinibatch: print("iMinibatch", iMinibatch))
                
                # Test
                model.eval()
                test_outputs = model(test_inputs)
                loss_function = torch.nn.BCELoss()
                test_loss = loss_function(test_outputs, test_targets)
                print("{:} loss: {:.4f}".format(model.get_label(), test_loss.item()))
                models.append(model)
            
            for model in models:
                print("{:} parameters: {:}".format(model.get_label(), [p.data.item() for p in model.parameters()]))
    
    if TEST_NETWORK_ACTIVATIONS:
        print("TEST_NETWORK_ACTIVATIONS")
        import training_data
        from scipy import linalg
        
        dataset_name = "B_PC1by75_N300_21-03-20"
        group_ids = [
            "gru11_b_2020-12-11",
            "elman11_b_2020-12-11",
            "grudiag11_b_2020-12-11"
        ]
        for group_id in group_ids:
            networks = training_data.load_models(group_id)
            inputs, targets, p_gens = data.load_test_data(dataset_name)
            epsilon = 1e-3
            n_networks = len(networks)
            n_time_steps = inputs.size(0)
            n_samples = inputs.size(1)
            
            hidden_state_delta_max = 0
            outputs_delta_max = 0
            for network in networks:
                n_units = network.hidden_layer.hidden_size
                w_out = network.output_layer.weight
                w_out_vector = w_out.squeeze()
                w_out_vector_normalized = w_out_vector / w_out_vector.norm(dim=0)
                perturbations_collinear_t = torch.empty(n_samples, n_units)
                perturbations_collinear_t[:, :] = w_out_vector_normalized.unsqueeze(0)
                perturbations_orthogonal_t_s = torch.empty(n_samples, n_units, (n_units - 1))
                ker_out = linalg.null_space(w_out.detach().numpy())
                perturbations_orthogonal_t_s[:, :, :] = torch.tensor(ker_out).unsqueeze(0)
                activations_pytorch = network.forward_activations(inputs)
                hidden_state_previous = activations_pytorch["hidden_state"].roll(+1, 0)
                hidden_state_previous[0, :, :] = 0.
                has_effect_at_tplus1 = False
                for t in range(0, n_time_steps, 7):#[0, 1, n_time_steps // 2, n_time_steps - 1]:
                    # Test: activations with manual and with pytorch computations should be epsilon-equal
                    activations_manual_t = network.compute_next_activations(inputs[t, :, :], hidden_state_previous[t, :, :])
                    hidden_state_manual_t = activations_manual_t["hidden_state"]
                    hidden_states_delta_t = torch.abs(activations_pytorch["hidden_state"][t, :, :] - hidden_state_manual_t)
                    outputs_delta_t = torch.abs(activations_pytorch["outputs"][t, :, :] - activations_manual_t["outputs"])
                    assert torch.all(hidden_states_delta_t < epsilon), \
                        "calculation error or imprecision (max-delta={:.4f})".format(hidden_states_delta_t.max())
                    assert torch.all(outputs_delta_t < epsilon), \
                        "calculation error or imprecision (max-delta={:.4f})".format(outputs_delta_t.max())
                    hidden_state_delta_max = max(hidden_state_delta_max, hidden_states_delta_t.max())
                    outputs_delta_max = max(outputs_delta_max, outputs_delta_t.max())
                    # Test: outputs with and without perturbations collinear to W_out should be different
                    # activations_perturbed_t = network.compute_next_activations(inputs[t, :, :],
                    #                                                            hidden_state_previous[t, :, :] +
                    #                                                            perturbations_collinear_t)
                    hidden_states_perturbed_t = hidden_state_manual_t + perturbations_collinear_t
                    outputs_perturbed_t = network.compute_outputs_from_hidden_state(hidden_states_perturbed_t)
                    outputs_perturbed_delta_t = torch.abs(outputs_perturbed_t - activations_manual_t["outputs"])
                    assert torch.all(outputs_perturbed_delta_t > epsilon), \
                        "effect of collinear perturbations should not be null (min-delta={:.4f})".format(outputs_perturbed_delta_t.min())
                    for i_orthogonal in range(n_units - 1):
                        # Test: outputs with and without perturbations orthogonal to W_out should be equal at time t
                        hidden_states_perturbed_t = hidden_state_manual_t + perturbations_orthogonal_t_s[:, :, i_orthogonal]
                        outputs_perturbed_t = network.compute_outputs_from_hidden_state(hidden_states_perturbed_t)
                        outputs_perturbed_delta_t = torch.abs(outputs_perturbed_t - activations_manual_t["outputs"])
                        assert torch.all(outputs_perturbed_delta_t < epsilon), \
                            "effect of orthogonal perturbations should be null at time t (max-delta={:.4f})".format(outputs_perturbed_delta_t.max())
                        # Test: outputs with and without perturbations orthogonal to W_out may not be equal at time t+1
                        hidden_states_perturbed_tminus1 = hidden_state_previous[t, :, :] + perturbations_orthogonal_t_s[:, :, i_orthogonal]
                        outputs_perturbed_t = network.compute_next_activations(inputs[t, :, :], hidden_states_perturbed_tminus1)["outputs"]
                        outputs_perturbed_delta_t = torch.abs(outputs_perturbed_t - activations_manual_t["outputs"])
                        has_effect_at_tplus1 = has_effect_at_tplus1 or torch.any(outputs_perturbed_delta_t > epsilon)
                assert has_effect_at_tplus1, \
                    "effect of orthogonal perturbations should not be always null at time t+1"
            print("group_id: ", group_id)
            print("hidden_state_delta_max", hidden_state_delta_max)
            print("outputs_delta_max", outputs_delta_max)
        
        
