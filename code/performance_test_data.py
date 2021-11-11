import argparse
import data
import measure
import numpy as np
import os
import pandas
import training_data

parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str, help="name of test dataset")
parser.add_argument("group_ids", type=str, nargs='+', help="list of trained model group ids")
parser.add_argument("-o", "--output", help="path to output for indidvidual data")
parser.add_argument('--do_output_stats', dest='do_output_stats', action='store_true')
parser.set_defaults(do_output_stats=False)
parser.add_argument('--include-bayes-uncoupled', dest='include_bayes_uncoupled', action='store_true')
parser.set_defaults(include_io_uncoupled=False)
parser.add_argument("-o-stats-path", "--output-stats-path", dest="output_stats_path",
    help="path to output for aggregate stats data")
args = parser.parse_args()
dataset_name = args.dataset_name
group_ids = args.group_ids
include_bayes_uncoupled = args.include_bayes_uncoupled
out_path = args.output
do_output_stats = args.do_output_stats
output_stats_path = args.output_stats_path

# get test data
test_data = data.load_test_data(dataset_name)
inputs, targets, p_gens = test_data
gen_process = data.load_gen_process(dataset_name)

n_groups = len(group_ids)
n_models = training_data.get_n_models(group_ids[0])
for group_id in group_ids[1:]:
    assert training_data.get_n_models(group_id) == n_models

# compute loss for each model
chance_loss = measure.get_chance_loss(inputs, targets)
io_loss = measure.get_io_loss(inputs, targets, gen_process)
loss_means = []
loss_ci_lower_s = []
loss_ci_upper_s = []
performance_means = []
performance_ci_lower_s = []
performance_ci_upper_s = []
i_model_median_performance_s = []
i_model_max_performance_s = []
losses_individual = np.empty((n_groups, n_models))
performances_individual = np.empty((n_groups, n_models))
group_ids_individual = [group_id for group_id in group_ids for _ in range(n_models)]
for i_group, group_id in enumerate(group_ids):
    models = training_data.load_models(group_id)
    losses = np.array([ measure.get_loss(model(inputs), targets) for model in models ])
    performances = measure.get_percent_value(losses, chance_loss, io_loss)
    loss_stats = measure.get_summary_stats(losses)
    performance_stats = measure.get_summary_stats(performances)
    i_median = np.argsort(-losses)[ (n_models + 1) // 2 ]
    i_max = np.argsort(losses)[0]
    loss_means.append(loss_stats["mean"])
    loss_ci_lower_s.append(loss_stats["ci_lower"])
    loss_ci_upper_s.append(loss_stats["ci_upper"])
    performance_means.append(performance_stats["mean"])
    performance_ci_lower_s.append(performance_stats["ci_lower"])
    performance_ci_upper_s.append(performance_stats["ci_upper"])
    i_model_median_performance_s.append(i_median)
    i_model_max_performance_s.append(i_max)
    losses_individual[i_group, :] = losses
    performances_individual[i_group, :] = performances
losses_individual = losses_individual.reshape(-1)
performances_individual = performances_individual.reshape(-1)

individual_data_dict = {
    "dataset_name": dataset_name,
    "chance_loss": chance_loss,
    "io_loss": io_loss,
    "group_id": group_ids_individual,
    "loss": losses_individual,
    "performance": performances_individual,
}
individual_df = pandas.DataFrame(individual_data_dict)
individual_df.to_csv(out_path)
print("Results saved at", out_path)

if do_output_stats:

    results_dict = { 
        "dataset_name": dataset_name,
        "chance_loss": chance_loss,
        "io_loss": io_loss,
        "group_id": group_ids,
        "loss_mean": loss_means,
        "loss_ci_lower": loss_ci_lower_s,
        "loss_ci_upper": loss_ci_upper_s,
        "performance_mean": performance_means,
        "performance_ci_lower": performance_ci_lower_s,
        "performance_ci_upper": performance_ci_upper_s,
        "i_model_median_performance": i_model_median_performance_s,
        "i_model_max_performance": i_model_max_performance_s
    }
    if include_bayes_uncoupled:
        import generate as gen
        import io_model
        gen_process_uncoupled = gen.GenerativeProcessMarkovIndependent(gen_process.p_change)
        io_uncoupled = io_model.IOModel(gen_process_uncoupled)
        results_dict["bayes_uncoupled_loss"] = measure.get_loss(io_uncoupled.get_predictions(inputs), targets)

    results_df = pandas.DataFrame(results_dict)
    results_df.to_csv(output_stats_path)
    print("Stats results saved at", output_stats_path)


