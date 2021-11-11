import os
import pandas
import pickle
import utils

DECODING_REGRESSION_STATS_FILENAME = "decoding_net_to_io_regression_stats.csv"
DECODING_REGRESSION_DATA_FILENAME = "decoding_net_to_io_regression_data.csv"

CONFIDENCE_PREDICTOR_ALL_KEYS = ["hidden_state", "hidden_gates_tplus1", "hidden_all"]
LOGODDS_PREDICTOR_KEY = "hidden_state"

KEY_LABEL_DICT = {
    "hidden_state": "Network hidden state",
    "hidden_gates_tplus1": "Network gates",
    "hidden_all": "Network hidden state\nand gates",
    "io_confidence_0": r"0 | 0",
    "io_confidence_1": r"1 | 1",
    "io_evidence_0": r"0 | 0",
    "io_evidence_1": r"1 | 1",
    "io_confidence": "Precision",
    "io_evidence": "Log odds"
}

PREDICTOR_LABEL_DICT = {
    "hidden_state": "Hidden state",
    "hidden_gates_tplus1": "Gates",
    "hidden_all": "Hidden state and gates",
}

def get_group_regression_stats_path(decoder_group_dir):
    return os.path.join(decoder_group_dir, DECODING_REGRESSION_STATS_FILENAME)

def get_group_regression_data_path(decoder_group_dir):
    return os.path.join(decoder_group_dir, DECODING_REGRESSION_DATA_FILENAME)
    
def load_decoders(decoder_group_dir, outcome_key, predictor_key, network_ids):
    return [ load_decoder(decoder_group_dir, outcome_key, predictor_key, network_id)
                for network_id in network_ids ]

def get_median_r2_network_id(decoder_group_dir, outcome_key, predictor_key):
    stats_path = get_group_regression_stats_path(decoder_group_dir)
    df = pandas.read_csv(stats_path)
    row = df.query("outcome == @outcome_key & predictor == @predictor_key").iloc[0]
    return row["r2_median_network_id"]

def get_decoder_fname(outcome_key, predictor_key, network_id):
    return "{}_{}_{}_decoder.pkl".format(outcome_key, predictor_key, network_id)

def get_decoder_path(decoder_group_dir, fname):
    return os.path.join(decoder_group_dir, fname)

def save_decoder(decoder_group_dir, decoder, fname):
    path = get_decoder_path(decoder_group_dir, fname)
    with open(path, 'wb') as file:
        pickle.dump(decoder, file)
        
def load_decoder(decoder_group_dir, outcome_key, predictor_key, network_id):
    fname = get_decoder_fname(outcome_key, predictor_key, network_id)
    path = get_decoder_path(decoder_group_dir, fname)
    with open(path, 'rb') as file:
        decoder = pickle.load(file)
    return decoder

def get_predictor_keys(decoder_group_dir, outcome_kind):
    if outcome_kind == "confidence":
        predictor_keys = get_confidence_predictor_keys(decoder_group_dir)
    elif outcome_kind.lower() == "logodds":
        predictor_keys = [LOGODDS_PREDICTOR_KEY]
    else:
        assert False, f"unknown outcome_kind: {outcome_kind}"
    return predictor_keys

def get_all_predictor_keys(outcome_kind):
    if outcome_kind == "confidence":
        predictor_keys = CONFIDENCE_PREDICTOR_ALL_KEYS
    elif outcome_kind.lower() == "logodds":
        predictor_keys = [LOGODDS_PREDICTOR_KEY]
    else:
        assert False, f"unknown outcome_kind: {outcome_kind}"
    return predictor_keys

def get_outcome_keys(decoder_group_dir, task_kind, outcome_kind):
    if task_kind.lower() == "markov":
        if outcome_kind.lower() == "confidence":
            outcome_keys = ["io_confidence_0", "io_confidence_1"]
        elif outcome_kind.lower() == "logodds":
            outcome_keys = ["io_evidence_0", "io_evidence_1"]
        else:
            assert False, f"unknown outcome_kind: {outcome_kind}"
    elif task_kind.lower() == "bernoulli":
        if outcome_kind.lower() == "confidence":
            outcome_keys = ["io_confidence"]
        elif outcome_kind.lower() == "logodds":
            outcome_keys = ["io_evidence"]
        else:
            assert False, f"unknown outcome_kind: {outcome_kind}"
    else:
        assert False, f"unknown task_kind: {task_kind}"
    return outcome_keys

def get_confidence_predictor_keys(decoder_group_dir):
    stats_path = get_group_regression_stats_path(decoder_group_dir)
    df = pandas.read_csv(stats_path)
    return utils.list_members_of_other_list(CONFIDENCE_PREDICTOR_ALL_KEYS, df["predictor"].unique())
