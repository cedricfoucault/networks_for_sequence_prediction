import argparse
import measure
import numpy as np
from scipy import stats
import utils
import pandas

# Compute two-tailed two independent sample t-test
# to compare regression slopes of two network groups

parser = argparse.ArgumentParser()
parser.add_argument("data_path_1", type=str)
parser.add_argument("data_path_2", type=str)
utils.add_arguments(parser, ["output"])
args = parser.parse_args()
data_path_1 = args.data_path_1
data_path_2 = args.data_path_2
out_path = args.output

df_1 = pandas.read_csv(data_path_1)
df_2 = pandas.read_csv(data_path_2)
slope_1 = df_1["slope"].to_numpy()
slope_2 = df_2["slope"].to_numpy()
tvalue, pvalue = stats.ttest_ind(slope_1, slope_2)

# Caclulate statistics on the slopes of the two groups separately
n_1 = len(slope_1)
n_2 = len(slope_2)
df = n_1 + n_2 - 2
summary_stats_1 = measure.get_summary_stats(slope_1)
summary_stats_2 = measure.get_summary_stats(slope_2)
mean_1 = summary_stats_1["mean"]
mean_2 = summary_stats_2["mean"]
ci_lower_1, ci_upper_1 = summary_stats_1["ci_lower"], summary_stats_1["ci_upper"]
ci_lower_2, ci_upper_2 = summary_stats_2["ci_lower"], summary_stats_2["ci_upper"]

# Calculate effect size (Cohen's d) from means and variances
var_1 = slope_1.var(ddof=1)
var_2 = slope_2.var(ddof=1)
pooled_std = np.sqrt(((n_1 - 1) * var_1 + (n_2 - 1) * var_2) / df)
cohen_d = (mean_1 - mean_2) / pooled_std
# Calculate effect size (Cohen's d) from the t statistic
# (this is a safety check, the two values of d should be roughly equal)
cohen_d_from_t = 2 * tvalue / np.sqrt(df)

df = pandas.Series(dict(
    data_path_1=data_path_1,
    data_path_2=data_path_2,
    tvalue=tvalue,
    pvalue=pvalue,
    cohen_d=cohen_d,
    cohen_d_from_t=cohen_d_from_t,
    mean_1=mean_1,
    ci_lower_1=ci_lower_1,
    ci_upper_1=ci_upper_1,
    mean_2=mean_2,
    ci_lower_2=ci_lower_2,
    ci_upper_2=ci_upper_2,
))
df.to_csv(out_path)
print("Results saved at", out_path)
