import argparse
import measure
import pandas
import utils

# Perform a two-way ANOVA of architecture * task on performance (via loss)

parser = argparse.ArgumentParser()
parser.add_argument("--data_paths", type=str, nargs="+")
parser.add_argument("--group_ids", type=str, nargs="+")
parser.add_argument("--architectures", type=str, nargs="+")
parser.add_argument("--tasks", type=str, nargs="+")
utils.add_arguments(parser, ["output"])
args = parser.parse_args()
data_paths = args.data_paths
group_ids = args.group_ids
architectures = args.architectures
tasks = args.tasks
out_path = args.output

n_args = len(data_paths)
assert len(group_ids) == n_args
assert len(architectures) == n_args
assert len(tasks) == n_args

# Build data to perform the two-way ANOVA on
anova_data_rows = []
for i, data_path in enumerate(data_paths):
    df = pandas.read_csv(data_path)
    group_id = group_ids[i]
    losses = df.query("group_id == @group_id")["loss"].to_numpy()
    rows = [ { "architecture": architectures[i], "task": tasks[i], "loss": loss } for loss in losses ]
    anova_data_rows.extend(rows)
anova_data = pandas.DataFrame(anova_data_rows)

anova_table = measure.stats_twoway_anova_table(anova_data,
    iv1_name="architecture", iv2_name="task", dv_name="loss")

anova_table.to_csv(out_path)
print("Results saved at", out_path)
