import argparse
import numpy as np
import pandas
import utils

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="path to csv file")
utils.add_arguments(parser, ["output"])
args = parser.parse_args()
data_path = args.data_path
out_path = args.output

df = pandas.read_csv(data_path)
aggregate_dict = {
    "data_dir": df["data_dir"].iloc[0],
    "hyperparameter_keys": [],
    "hyperparameter_values": [],
    "n_trials": [],
    "did_all_trial_complete": [],
    "mean_loss": [],
    "min_loss": [],
}
hyperparameter_value_combinations = df["hyperparameter_values"].unique()
for value_combination in hyperparameter_value_combinations:
    dfv = df[df["hyperparameter_values"] == value_combination]
    n_trials = len(dfv)
    did_all_trial_complete = np.all(dfv["did_trial_complete"])
    losses = dfv["validation_loss"]
    loss_mean = losses.mean()
    loss_min = losses.min()
    aggregate_dict["hyperparameter_keys"].append(dfv["hyperparameter_keys"].iloc[0])
    aggregate_dict["hyperparameter_values"].append(value_combination)
    aggregate_dict["n_trials"].append(n_trials)
    aggregate_dict["did_all_trial_complete"].append(did_all_trial_complete)
    aggregate_dict["mean_loss"].append(loss_mean)
    aggregate_dict["min_loss"].append(loss_min)

aggregate_df = pandas.DataFrame(aggregate_dict)
aggregate_df.to_csv(out_path)
print("Results saved at", out_path)
