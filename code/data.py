import generate as gen
import os
import torch

DATA_DIR_RELATIVE_PATH = "../datasets" # directory location relative to this .py file
DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), DATA_DIR_RELATIVE_PATH)

## Training data set
TRAIN_N_MINIBATCHES = 80
TRAIN_MINIBATCH_N_SEQUENCES = 20

## Testing data set
TEST_N_SEQUENCES = 200

# Generate datasets
def generate_minibatches_data(gen_process, n_minibatches, n_sequences):
    minibatches_inputs = []
    minibatches_targets = []
    minibatches_p_gens = []
    for _ in range(n_minibatches):
        inputs, targets, p_gens = gen.generate_netinput_target_probagen(gen_process, n_sequences)
        minibatches_inputs.append(inputs)
        minibatches_targets.append(targets)
        minibatches_p_gens.append(p_gens)
    return (minibatches_inputs, minibatches_targets, minibatches_p_gens)

def generate_interleaved_minibatches_data(gen_process_1, gen_process_2, n_minibatches, n_sequences):
    minibatches_inputs = []
    minibatches_targets = []
    minibatches_p_gens = []
    for i_mb in range(n_minibatches):
        gen_process = gen_process_1 if i_mb % 2 == 0 else gen_process_2
        inputs, targets, p_gens = gen.generate_netinput_target_probagen(gen_process, n_sequences)
        minibatches_inputs.append(inputs)
        minibatches_targets.append(targets)
        minibatches_p_gens.append(p_gens)
    return (minibatches_inputs, minibatches_targets, minibatches_p_gens)

def generate_train_data(gen_process, train_n_minibatches=TRAIN_N_MINIBATCHES,
train_minibatch_n_sequences=TRAIN_MINIBATCH_N_SEQUENCES):
    train_data = generate_minibatches_data(gen_process, train_n_minibatches, train_minibatch_n_sequences)
    return train_data

def generate_interleaved_train_data(gen_process_1, gen_process_2,
                                    train_n_minibatches=TRAIN_N_MINIBATCHES,
                                    train_minibatch_n_sequences=TRAIN_MINIBATCH_N_SEQUENCES):
    train_data = generate_interleaved_minibatches_data(gen_process_1, gen_process_2,
                                                       train_n_minibatches, train_minibatch_n_sequences)
    return train_data

def generate_test_data(gen_process, test_n_sequences=TEST_N_SEQUENCES):
    test_data = gen.generate_netinput_target_probagen(gen_process, test_n_sequences)
    return test_data

def generate_mixed_test_data(gen_process_1, gen_process_2, test_n_sequences=TEST_N_SEQUENCES):
    inputs_1, targets_1, p_gens_1 = gen.generate_netinput_target_probagen(gen_process_1, test_n_sequences // 2)
    inputs_2, targets_2, p_gens_2 = gen.generate_netinput_target_probagen(gen_process_2, test_n_sequences - test_n_sequences // 2)
    inputs = torch.cat((inputs_1, inputs_2), dim=1)
    targets = torch.cat((targets_1, targets_2), dim=1)
    p_gens = torch.cat((p_gens_1, p_gens_2), dim=1)
    return inputs, targets, p_gens

# Seralize data
def get_data_dict_from_tuple(data):
    inputs, targets, p_gens = data
    data_dict = {
            "inputs": inputs,
            "targets": targets,
            "p_gens": p_gens
    }
    return data_dict

def get_data_tuple_from_dict(data_dict):
    inputs = data_dict["inputs"]
    targets = data_dict["targets"]
    p_gens = data_dict["p_gens"]
    return inputs, targets, p_gens

# Save, Load and Delete Datasets
def get_data_dict_path(fbasename):
    return os.path.join(DATA_DIR, "{:}_dict.pt".format(fbasename))

def get_gen_process_dict_path(fbasename):
    return os.path.join(DATA_DIR, "{:}_gen_process_dict.pt".format(fbasename))

def get_io_outputs_path(fbasename):
    return os.path.join(DATA_DIR, "{:}_io_outputs.pt".format(fbasename))

def save_data(data, fbasename):
    data_dict = get_data_dict_from_tuple(data)
    f = get_data_dict_path(fbasename)
    torch.save(data_dict, f)
    
def load_data(fbasename):
    f = get_data_dict_path(fbasename)
    data_dict = torch.load(f)
    return get_data_tuple_from_dict(data_dict)

def delete_data(fbasename):
    f = get_data_dict_path(fbasename)
    if os.path.isfile(f):
        os.remove(f)

def get_test_name(fbasename):
    return "{:}_test".format(fbasename)

def get_train_name(fbasename):
    return "{:}_train".format(fbasename)

def save_train_data(gen_process, train_data, fbasename):
    train_fbasename = get_train_name(fbasename)
    save_data(train_data, train_fbasename)
    save_gen_process(gen_process, fbasename)

def save_test_data(test_data, fbasename):
    test_fbasename = get_test_name(fbasename)
    save_data(test_data, test_fbasename)

def load_train_test_data(fbasename, train_n_minibatches=None):
    train_data, gen_process = load_train_data(fbasename, train_n_minibatches)
    test_data = load_test_data(fbasename)
    return train_data, test_data, gen_process

def load_train_data(fbasename, train_n_minibatches=None):
    train_fbasename = get_train_name(fbasename)
    train_data = load_data(train_fbasename)
    gen_process = load_gen_process(fbasename)
    return train_data, gen_process

def load_test_data(fbasename):
    test_fbasename = get_test_name(fbasename)
    test_data = load_data(test_fbasename)
    return test_data
    
def delete_train_test_data(fbasename):
    train_fbasename = get_train_name(fbasename)
    test_fbasename = get_test_name(fbasename)
    delete_data(train_fbasename)
    delete_data(test_fbasename)
    delete_gen_process(fbasename)

def save_gen_process(gen_process, fbasename):
    f_gen_process = get_gen_process_dict_path(fbasename)
    gen_process_dict = gen.get_dict_from_gen_process(gen_process)
    torch.save(gen_process_dict, f_gen_process)

def load_gen_process(fbasename):
    f_gen_process = get_gen_process_dict_path(fbasename)
    gen_process_dict = torch.load(f_gen_process)
    return gen.get_gen_process_from_dict(gen_process_dict)
    
def delete_gen_process(fbasename):
    f_gen_process = get_gen_process_dict_path(fbasename)
    if os.path.isfile(f_gen_process):
        os.remove(f_gen_process)
        
def save_io_outputs(io_outputs, fbasename):
    fpath = get_io_outputs_path(fbasename)
    torch.save(io_outputs, fpath)

def load_io_outputs(fbasename):
    fpath = get_io_outputs_path(fbasename)
    return torch.load(fpath)

def get_streak_data_path(fbasename):
    return os.path.join(DATA_DIR, f"{fbasename}.pt")

def save_streak_data(streak_data, fbasename):
    fpath = get_streak_data_path(fbasename)
    torch.save(streak_data, fpath)

def load_streak_data(fbasename):
    fpath = get_streak_data_path(fbasename)
    return torch.load(fpath)


