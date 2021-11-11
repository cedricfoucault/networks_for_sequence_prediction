import argparse
import data
import measure
import model
import numpy as np
import os
import pandas
from ray import tune
import utils

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str,help="path to the directory containing the ray data")
utils.add_arguments(parser, ["output"])
args = parser.parse_args()
data_dir = args.data_dir
out_path = args.output

analysis = tune.Analysis(data_dir)
df = analysis.dataframe()
df = df[df["n_training_updates"] == df["n_training_updates"].max()] # filter out trials that did not complete

# compute io loss on validation dataset
validate_dataset_name = df["config/validate_dataset_name"][0]
inputs, targets, p_gens = data.load_test_data(validate_dataset_name)
gen_process = data.load_gen_process(validate_dataset_name)
io_loss = measure.get_io_loss(inputs, targets, gen_process)
chance_loss = measure.get_chance_loss(inputs, targets)

# compute performance from validation losses
n_units_col = "config/n_units"
loss_col = "validation_loss"
n_units_values = np.sort(df[n_units_col].unique())
n_values = len(n_units_values)
n_trained_parameters_values = np.empty((n_values))
mean_performances = np.empty((n_values))
ci_upper_performances = np.empty((n_values))
ci_lower_performances = np.empty((n_values))
max_performances = np.empty((n_values))
for i, n_units in enumerate(n_units_values):
    df_rows = df[df[n_units_col] == n_units]
    losses = df_rows[loss_col].to_numpy()
    performances = measure.get_percent_value(losses, chance_loss, io_loss)
    performance_stats = measure.get_summary_stats(performances)
    mean_performances[i] = performance_stats["mean"]
    ci_upper_performances[i] = performance_stats["ci_upper"]
    ci_lower_performances[i] = performance_stats["ci_lower"]
    max_performances[i] = performances.max()
    logdir = df_rows["logdir"].iloc[0]
    sample_network_path = os.path.join(logdir, "trained_model.pt")
    n_trained_parameters_values[i] = model.Network.load(sample_network_path).get_n_trainable_parameters()


data_dict = {
    "validate_dataset_name": validate_dataset_name,
    "sample_network_path": sample_network_path,
    "n_units": n_units_values,
    "n_trained_parameters": n_trained_parameters_values,
    "mean_performance": mean_performances,
    "ci_upper_performance": ci_upper_performances,
    "ci_lower_performance": ci_lower_performances,
    "max_performance": max_performances,
}
df = pandas.DataFrame(data_dict)
df.to_csv(out_path)
print("Results saved at", out_path)