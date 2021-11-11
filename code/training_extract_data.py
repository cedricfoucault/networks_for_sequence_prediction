import argparse
import model
import os
import pandas
from ray import tune
import shutil
import training_data

# Load input experiment dataframe
parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str,help="path to the directory containing the ray data")
parser.add_argument("group_id", type=str, help="trained model group id")
parser.add_argument("-o", "--output", help="path to output directory")
args = parser.parse_args()
in_dir = args.data_dir
analysis = tune.Analysis(in_dir)
df = analysis.dataframe()

# Create directory to save output data to
group_id = args.group_id
out_dir = training_data.get_group_directory_path(group_id)
assert os.path.realpath(args.output) == os.path.realpath(out_dir), \
    f"Inconsistent paths {os.path.realpath(out_dir)} and argument {os.path.realpath(args.output)}"
try:
    os.mkdir(out_dir)
except FileExistsError:
    # Remove directory and all its contents and recreate it
    import shutil
    shutil.rmtree(out_dir)
    os.mkdir(out_dir)

# Extract data for each training run
network_ids = df["trial_id"].to_numpy()
network_paths = []
progress_paths = []
for network_id in network_ids:
    logdir = df[df["trial_id"] == network_id]["logdir"].iloc[0]
    # load and save trained networks
    in_network_path = os.path.join(logdir, "trained_model.pt")
    if "config/model_class" in df:
        model_class_str = df["config/model_class"].iloc[0]
    else:
        model_class_str = "network"
    model_class = model.get_class_from_string(model_class_str)
    network = model_class.load(in_network_path)
    network_path = training_data.get_network_path(group_id, network_id)
    network.save(network_path)
    network_paths.append(network_path)
    # load training progress data
    in_progress_path = os.path.join(logdir, "progress.csv")
    in_progress_df = pandas.read_csv(in_progress_path)
    in_progress_df.sort_values("n_training_updates", ascending=True, inplace=True)
    n_training_updates = in_progress_df["n_training_updates"].to_numpy()
    validation_loss = in_progress_df["validation_loss"].to_numpy()
    # save training progress data as a new dataframe
    ## - network id
    ## - network file path
    ## - number of training updates
    ## - validation loss
    progress_dict = {
            "network_id": network_id,
            "network_path": network_path,
            "n_training_updates": n_training_updates,
            "validation_loss": validation_loss
    }
    progress_dataframe = pandas.DataFrame(progress_dict)
    progress_path = training_data.get_progress_path(group_id, network_id)
    progress_dataframe.to_csv(progress_path)
    progress_paths.append(progress_path)
    # save parameters through training if available
    in_parameters_through_training_path = os.path.join(logdir, "parameters_through_training.pt")
    if os.path.isfile(in_parameters_through_training_path):
        out_parameters_through_training_path = \
            training_data.get_parameters_through_training_path(group_id, network_id)
        shutil.copyfile(in_parameters_through_training_path,
            out_parameters_through_training_path)


def add_first_value_if_exists(d, df, key):
    if key in df:
        d[key] = df[key].iloc[0]

# Extract metadata (for all run) and save as  a new data frame to save:
## - number of networks
## - network ids
## - network paths
## - training progress paths
## - date
## - config/seed
## - config/training_dataset
## - config/validation_dataset
## - config/unit_type
## - config/n_units
## - config/optimizer.name
## - config/optimizer.lr
metadata_dict = {
        "group_id": group_id,
        "n_networks": len(network_ids),
        "network_id": network_ids,
        "network_path": network_paths,
        "progress_path": progress_paths,
        "date": df["date"].iloc[0],
        "config/seed": df["config/seed"].iloc[0],
        }
add_first_value_if_exists(metadata_dict, df, "config/train_dataset_name")
add_first_value_if_exists(metadata_dict, df, "config/validate_dataset_name")
add_first_value_if_exists(metadata_dict, df, "config/optimizer.name")
add_first_value_if_exists(metadata_dict, df, "config/optimizer.lr")
add_first_value_if_exists(metadata_dict, df, "config/unit_type")
add_first_value_if_exists(metadata_dict, df, "config/n_units"),
add_first_value_if_exists(metadata_dict, df, "config/n_layers"),
add_first_value_if_exists(metadata_dict, df, "config/shuffle_minibatches")
add_first_value_if_exists(metadata_dict, df, "config/model_class")
add_first_value_if_exists(metadata_dict, df, "config/estimate_type")
add_first_value_if_exists(metadata_dict, df, "config/no_train")
metadata_dataframe = pandas.DataFrame(metadata_dict)
metadata_path = training_data.get_metadata_path(group_id)
metadata_dataframe.to_csv(metadata_path)

print("data saved at", out_dir)
