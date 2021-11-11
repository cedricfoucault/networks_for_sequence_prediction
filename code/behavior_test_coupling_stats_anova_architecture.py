import argparse
from behavior_test_coupling_common import get_changes_in_prediction_by_coupled
import measure
import pandas
import utils

# Perform a two-way ANOVA of architecture * training environment on the change in prediction

parser = argparse.ArgumentParser()
parser.add_argument("--data_paths", type=str, nargs="+")
parser.add_argument("--architectures", type=str, nargs="+")
utils.add_arguments(parser, ["output"])
args = parser.parse_args()
data_paths = args.data_paths
architectures = args.architectures
out_path = args.output

n_args = len(data_paths)
assert len(architectures) == n_args

# Build data to perform the two-way ANOVA on
anova_data_rows = []
for i, data_path in enumerate(data_paths):
    df = pandas.read_csv(data_path)
    changes_in_prediction_by_coupled = get_changes_in_prediction_by_coupled(df)
    for coupled in [True, False]:
        changes_in_prediction = changes_in_prediction_by_coupled[coupled]
        changes_in_prediction = changes_in_prediction.reshape(-1) # flatten over networks and streak lengths
        rows = [ 
            { "architecture": architectures[i], "coupled": coupled, "change_in_prediction": change } \
            for change in changes_in_prediction
        ]
        anova_data_rows.extend(rows)
anova_data = pandas.DataFrame(anova_data_rows)

anova_table = measure.stats_twoway_anova_table(anova_data,
    iv1_name="architecture", iv2_name="coupled", dv_name="change_in_prediction")

anova_table.to_csv(out_path)
print("Results saved at", out_path)
