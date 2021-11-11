import argparse
import numpy as np
import measure
import pandas
import scipy.stats as stats
import utils

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="individual data path")
parser.add_argument("group_ids", nargs="*", help="group ids to compare")
utils.add_arguments(parser, ["output"])
args = parser.parse_args()
data_path = args.data_path
group_ids = args.group_ids
out_path = args.output

df = pandas.read_csv(data_path)
group_ids = utils.list_members_of_other_list(group_ids, df["group_id"].unique())
n_groups = len(group_ids)

chance_loss = df["chance_loss"].iloc[0]
io_loss = df["io_loss"].iloc[0]

performances_by_group = []
mean_performance_by_group = []
for group_id in group_ids:
    losses = df.query("group_id == @group_id")["loss"].to_numpy()
    performances = measure.get_percent_value(losses, chance_loss, io_loss)
    performances_by_group.append(performances)
    mean_performance_by_group.append(performances.mean())

# order groups by decreasing performance i.e. increasing loss
indices_decreasing_perf = np.argsort(-np.array(mean_performance_by_group))

# test all pairwise comparisons
group_ids_higher = []
group_ids_lower = []
stat_labels = []
t_values = []
p_values = []
performance_mean_highers = []
performance_mean_lowers = []
performance_mean_differences = []
performance_mean_difference_welch_t_intervals = []
for i_1 in range(n_groups):
    i_higher = indices_decreasing_perf[i_1]
    for i_2 in range(i_1+1, n_groups):
        i_lower = indices_decreasing_perf[i_2]
        performances_higher = performances_by_group[i_higher]
        performances_lower = performances_by_group[i_lower]
        t_val, p_val = stats.ttest_ind(
            performances_higher,
            performances_lower,
            equal_var=False
        )
        performance_mean_higher = performances_higher.mean()
        performance_mean_lower = performances_lower.mean()
        performance_mean_difference = performance_mean_higher - performance_mean_lower
        # Welch's t-interval formula (to estimate a confidence interval for the mean difference)
        # see https://online.stat.psu.edu/stat415/lesson/3/3.2
        n_1 = len(performances_higher)
        n_2 = len(performances_lower)
        var_1 = performances_higher.var(ddof=1)
        var_2 = performances_lower.var(ddof=1)
        var_over_n_1 = var_1 / n_1
        var_over_n_2 = var_2 / n_2
        welch_df = (var_over_n_1 + var_over_n_2) ** 2 / \
                   (var_over_n_1 ** 2 / (n_1 - 1) + var_over_n_2 ** 2 / (n_2 - 1))
        t_crit = stats.t.ppf(1 - 0.025, welch_df)
        difference_welch_t_interval = t_crit * np.sqrt(var_over_n_1 + var_over_n_2)
        group_ids_higher.append(group_ids[i_higher])
        group_ids_lower.append(group_ids[i_lower])
        t_values.append(t_val)
        p_values.append(p_val)
        stat_labels.append(utils.stat_label(p_val))
        performance_mean_highers.append(performance_mean_higher)
        performance_mean_lowers.append(performance_mean_lower)
        performance_mean_differences.append(performance_mean_difference)
        performance_mean_difference_welch_t_intervals.append(difference_welch_t_interval)

data_dict = {
    "group_id_higher": group_ids_higher,
    "group_id_lower": group_ids_lower,
    "stat_label": stat_labels,
    "t_value": t_values,
    "p_value": p_values,
    "performance_mean_higher": performance_mean_highers,
    "performance_mean_lower": performance_mean_lowers,
    "performance_mean_difference": performance_mean_differences,
    "performance_mean_difference_welch_t_interval": performance_mean_difference_welch_t_intervals,
}
df = pandas.DataFrame(data_dict)
df.to_csv(out_path)
print("Results saved at", out_path)

