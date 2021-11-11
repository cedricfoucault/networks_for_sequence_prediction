import argparse
import pandas
import hyperparam_config as hconfig
from ray import tune
import utils

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str, help="path to the directory containing the ray data")
utils.add_arguments(parser, ["output"])
args = parser.parse_args()
data_dir = args.data_dir
out_path = args.output

analysis = tune.Analysis(data_dir)
df = analysis.dataframe()

hyperparameter_columns_possible = [
        "config/" + hconfig.OPTIMIZER_LR_KEY,
        "config/" + hconfig.INITIALIZATION_WEIGHT_INPUT_TO_HIDDEN_STD_KEY,
        "config/" + hconfig.INITIALIZATION_WEIGHT_HIDDEN_TO_HIDDEN_STD_KEY,
        "config/" + hconfig.INIT_DIAGONAL_MEAN_KEY,
]
hyperparameter_columns = utils.list_members_of_other_list(hyperparameter_columns_possible, df.columns)
n_training_updates_max = df["n_training_updates"].max()
df["hyperparameter_values"] = list(zip(*[df[col] for col in hyperparameter_columns]))
df["hyperparameter_keys"] = [tuple([col[len("config/"):] for col in hyperparameter_columns])] * len(df)
df["did_trial_complete"] = df["n_training_updates"] == n_training_updates_max
df["data_dir"] = data_dir

df_new = df[["data_dir", "hyperparameter_keys", "hyperparameter_values", "did_trial_complete", "validation_loss"]].copy()

df_new.to_csv(out_path)
print("Results saved at", out_path)
