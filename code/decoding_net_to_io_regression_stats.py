import argparse
import decoding_data
import measure
import numpy as np
import pandas

parser = argparse.ArgumentParser()
parser.add_argument("decoder_group_dir", type=str, help="path to decoder group directory")
args = parser.parse_args()
decoder_group_dir = args.decoder_group_dir

data_path = decoding_data.get_group_regression_data_path(decoder_group_dir)
df = pandas.read_csv(data_path)

predictors = df["predictor"].unique()
outcomes = df["outcome"].unique()
network_ids = df["network_id"].unique()
n_networks = len(network_ids)
group_id = df["group_id"].iloc[0]

out_predictors = []
out_outcomes = []
r2_means = []
r2_mean_ci_uppers = []
r2_mean_ci_lowers = []
r2_maxs = []
r2_medians = []
r2_median_network_ids = []
r2_median_ci_uppers = []
r2_median_ci_lowers = []
r_means = []
r_mean_ci_uppers = []
r_mean_ci_lowers = []
r_maxs = []
r_medians = []
r_median_ci_uppers = []
r_median_ci_lowers = []
for predictor in predictors:
    for outcome in outcomes:
        r2_values = np.empty((n_networks))
        r_values = np.empty((n_networks))
        for i_network, network_id in enumerate(network_ids):
            dfq = df.query("(predictor == @predictor) &\
                    (outcome == @outcome) &\
                    (network_id == @network_id)")
            assert len(dfq) == 1, "More than one entry in dataframe matching query: {:d}".format(len(dfq))
            r2_values[i_network] = dfq["r2"].iloc[0]
            r_values[i_network] = dfq["r"].iloc[0]
        r2_stats = measure.get_summary_stats(r2_values)
        r2_median = np.median(r2_values)
        r2_median_index = np.argsort(r2_values)[(n_networks + 1) // 2]
        r2_median_network_id = network_ids[r2_median_index]
        r2_median_ci_lower, r2_median_ci_upper = measure.get_median_ci(r2_values)
        r_stats = measure.get_summary_stats(r_values)
        r_median = np.median(r_values)
        r_median_ci_lower, r_median_ci_upper = measure.get_median_ci(r_values)
        # append row to dataframe
        out_predictors.append(predictor)
        out_outcomes.append(outcome)
        r2_means.append(r2_stats["mean"])
        r2_maxs.append(r2_values.max())
        r2_mean_ci_lowers.append(r2_stats["ci_lower"])
        r2_mean_ci_uppers.append(r2_stats["ci_upper"])
        r2_medians.append(r2_median)
        r2_median_network_ids.append(r2_median_network_id)
        r2_median_ci_lowers.append(r2_median_ci_lower)
        r2_median_ci_uppers.append(r2_median_ci_upper)
        r_means.append(r_stats["mean"])
        r_maxs.append(r_values.max())
        r_mean_ci_lowers.append(r_stats["ci_lower"])
        r_mean_ci_uppers.append(r_stats["ci_upper"])
        r_medians.append(r_median)
        r_median_ci_lowers.append(r_median_ci_lower)
        r_median_ci_uppers.append(r_median_ci_upper)

out_df_dict = dict(
    group_id=group_id,
    predictor=out_predictors,
    outcome=out_outcomes,
    r2_mean=r2_means,
    r2_mean_ci_lower=r2_mean_ci_lowers,
    r2_mean_ci_upper=r2_mean_ci_uppers,
    r2_max=r2_maxs,
    r2_median=r2_medians,
    r2_median_network_id=r2_median_network_ids,
    r2_median_ci_lower=r2_median_ci_lowers,
    r2_median_ci_upper=r2_median_ci_uppers,
    r_mean=r_means,
    r_mean_ci_lower=r_mean_ci_lowers,
    r_mean_ci_upper=r_mean_ci_uppers,
    r_max=r_maxs,
    r_median=r_medians,
    r_median_ci_lower=r_median_ci_lowers,
    r_median_ci_upper=r_median_ci_uppers,
)
out_df = pandas.DataFrame(out_df_dict)
out_path = decoding_data.get_group_regression_stats_path(decoder_group_dir)
out_df.to_csv(out_path)
print("Data saved at", out_path)

