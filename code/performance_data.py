import pandas
import training_data

def get_median_performance_model(group_id, performance_stats_path):
    df_perf_stats = pandas.read_csv(performance_stats_path)
    assert group_id in df_perf_stats["group_id"].unique(), \
        "group_id not found in given stats file"
    index = df_perf_stats[df_perf_stats["group_id"] == group_id]["i_model_median_performance"].iloc[0]
    return training_data.load_models(group_id)[index]
