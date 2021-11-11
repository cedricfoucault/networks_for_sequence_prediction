import argparse
import numpy as np
from scipy import stats
import utils
import pandas

# Compute linear regression delta_lr = f(delta_conf) for each network

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="perturbation experiment data path")
utils.add_arguments(parser, ["output"])
args = parser.parse_args()
data_path = args.data_path
out_path = args.output

data = np.load(data_path)
delta_confs = data["delta_confs"]
delta_lrs = data["delta_lrs"]
n_delta_confs = delta_lrs.shape[0]
n_networks = delta_lrs.shape[1]
assert n_delta_confs == delta_confs.shape[0]
delta_lr_means = delta_lrs.mean(axis=2)

df_rows = []
for i_network in range(n_networks):
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(delta_confs, delta_lr_means[:, i_network])
    df_rows.append(dict(data_path=data_path, slope=slope, intercept=intercept))

df = pandas.DataFrame(df_rows)
df.to_csv(out_path)
print("Results saved at", out_path)
