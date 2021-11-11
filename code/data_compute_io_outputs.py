import argparse
import data
import io_model

parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str, help="name of dataset of sequences to compute io outputs on")
args = parser.parse_args()
dataset_name = args.dataset_name

test_data = data.load_test_data(dataset_name)
inputs, targets, p_gens = test_data
gen_process = data.load_gen_process(dataset_name)

io = io_model.IOModel(gen_process)
io_output_stats = io.get_output_stats(inputs)
data.save_io_outputs(io_output_stats, dataset_name)
