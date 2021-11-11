import argparse
import numpy as np
import pandas

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="path to raw data file")
parser.add_argument("-o", "--output", help="path to output file")
args = parser.parse_args()
data_path = args.data_path
out_path = args.output

df = pandas.read_csv(data_path)

dataset_name = df["dataset_name"].iloc[0]
io_pearsonr = df["io_pearsonr"].iloc[0]

group_ids = df["group_id"].unique()

data_group_ids = []
data_predictor_keys = []
data_pearsonr_means = []
data_pearsonr_medians = []
data_pearsonr_maxs = []
data_pearsonr_mins = []
for group_id in group_ids:
    dfg = df.query("(group_id == @group_id)")
    predictor_keys = dfg["predictor_key"].unique()
    for predictor_key in predictor_keys:
        dfgp = dfg.query("(predictor_key == @predictor_key)")
        pearsonrs = dfgp["pearsonr"].to_numpy()
        data_group_ids.append(group_id)
        data_predictor_keys.append(predictor_key)
        data_pearsonr_means.append(pearsonrs.mean())
        data_pearsonr_medians.append(np.median(pearsonrs))
        data_pearsonr_maxs.append(pearsonrs.max())
        data_pearsonr_mins.append(pearsonrs.min())

data_dict = {
    "dataset_name": dataset_name,
    "io_pearsonr": io_pearsonr,
    "group_id": data_group_ids,
    "predictor_key": data_predictor_keys,
    "pearsonr_mean": data_pearsonr_means,
    "pearsonr_median": data_pearsonr_medians,
    "pearsonr_max": data_pearsonr_maxs,
    "pearsonr_min": data_pearsonr_mins,
}
df = pandas.DataFrame(data_dict)
df.to_csv(out_path)
print("Results saved at", out_path)
