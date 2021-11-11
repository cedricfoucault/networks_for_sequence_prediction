import data
import model
import os
import pandas
import torch

TRAINING_DATA_DIR_RELATIVE_PATH = "../trained_models/agents" # directory location relative to this .py file
TRAINING_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), TRAINING_DATA_DIR_RELATIVE_PATH)

def get_group_directory_path(group_id):
    return os.path.join(TRAINING_DATA_DIR, group_id)

def get_metadata_path(group_id):
    group_dir = get_group_directory_path(group_id)
    return os.path.join(group_dir, "metadata.csv")

def get_network_path(group_id, network_id):
    group_dir = get_group_directory_path(group_id)
    return os.path.join(group_dir, "network_{}.pt".format(network_id))

def get_progress_path(group_id, network_id):
    group_dir = get_group_directory_path(group_id)
    return os.path.join(group_dir, "progress_{}.csv".format(network_id))

def get_parameters_through_training_path(group_id, network_id):
    group_dir = get_group_directory_path(group_id)
    return os.path.join(group_dir, "parameters_through_training_{}.pt".format(network_id))

def load_metadata(group_id):
    metadata_df = pandas.read_csv(get_metadata_path(group_id))
    return metadata_df

def get_n_models(group_id):
    return load_metadata(group_id)["n_networks"].iloc[0]

def load_models(group_id):
    models , _ = load_models_ids(group_id)
    return models

def load_model_ids(group_id):
    metadata_df = load_metadata(group_id)
    return metadata_df["network_id"].unique()

def load_models_ids(group_id):
    model_class = load_model_class(group_id)
    model_ids = load_model_ids(group_id)
    models = [ model_class.load(get_network_path(group_id, model_id)) for model_id in model_ids ]
    return models, model_ids

def load_model_with_id(group_id, model_id):
    model_class = load_model_class(group_id)
    return model_class.load(get_network_path(group_id, model_id))

def load_parameters_through_training_with_id(group_id, model_id):
    path = get_parameters_through_training_path(group_id, model_id)
    return torch.load(path)

def load_model_class(group_id):
    metadata_df = load_metadata(group_id)
    if "config/model_class" in metadata_df:
        model_class_str = metadata_df["config/model_class"].iloc[0]
    else:
        model_class_str = "network"
    return model.get_class_from_string(model_class_str)

def load_progress_dataframe(group_id, network_id):
    return pandas.read_csv(get_progress_path(group_id, network_id))

def load_gen_process(group_id):
    metadata_df = load_metadata(group_id)
    train_dataset_name = metadata_df["config/train_dataset_name"].iloc[0]
    return data.load_gen_process(train_dataset_name)
    