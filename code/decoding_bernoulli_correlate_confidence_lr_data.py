import argparse
import data
import decoding_data
import pandas
import io_model
import measure
import scipy.stats as stats
import training_data

def get_lr_tplus1(predictions, inputs):
    lrs_tplus1 = measure.get_learning_rate(predictions, inputs)[1:, :, :]
    return lrs_tplus1.reshape(-1).detach().numpy()

def get_confidences_t_lr_tplus1(inputs, network, decoder, predictor_key):
    activations = network.forward_activations(inputs)
    predictor_tensor_dict = measure.get_predictor_tensor_dict_from_activations(activations)
    confidences_t = measure.get_decoded_tensor(decoder, predictor_tensor_dict[predictor_key])
    confidence_t = confidences_t.reshape(-1).detach().numpy()
    lr_tplus1 = get_lr_tplus1(activations["outputs"], inputs)
    return confidence_t, lr_tplus1

parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str, help="name of test dataset")
parser.add_argument("--group_ids", nargs="+", type=str, help="trained model group ids")
parser.add_argument("--decoder_group_dirs", nargs="+", type=str, help="path to decoders' group directory")
parser.add_argument("-o", "--output", help="path to output file")
args = parser.parse_args()
dataset_name = args.dataset_name
group_ids = args.group_ids
decoder_group_dirs = args.decoder_group_dirs
out_path = args.output

outcome_key = "io_confidence"

n_groups = len(group_ids)
assert n_groups == len(decoder_group_dirs), "number of group ids in decoders don't match"

inputs, _ , _ = data.load_test_data(dataset_name)
gen_process = data.load_gen_process(dataset_name)

# Calculate pearson correlation of confidence(t) and learning rate (t+1) of ideal observer
io = io_model.IOModel(gen_process)
io_outputs = io.get_output_stats(inputs)
io_confidences_t = measure.get_confidence_from_sds(io_outputs["sd"])[:-1, :, :]
io_confidence_t = io_confidences_t.reshape(-1).numpy()
io_lr_tplus1 = get_lr_tplus1(io_outputs["mean"], inputs)
io_pearsonr, io_pearsonr_p = stats.pearsonr(io_confidence_t, io_lr_tplus1)

# Calculate pearson correlation of confidence(t) and learning rate (t+1)
# for each group/predictor/model, and aggregate
data_group_ids = []
data_predictor_keys = []
data_model_ids = []
data_pearsonrs = []
data_pearsonr_ps = []
for i_group, group_id in enumerate(group_ids):
    decoder_group_dir = decoder_group_dirs[i_group]
    networks, network_ids = training_data.load_models_ids(group_id)
    predictor_keys = decoding_data.get_confidence_predictor_keys(decoder_group_dir)
    for predictor_key in predictor_keys:
        decoders = decoding_data.load_decoders(decoder_group_dir, outcome_key, predictor_key, network_ids)
        for i_network, network in enumerate(networks):
            model_id = network_ids[i_network]
            decoder = decoders[i_network]
            confidence_t, lr_tplus1 = get_confidences_t_lr_tplus1(inputs, network, decoder, predictor_key)
            pearsonr, pearsonr_p = stats.pearsonr(confidence_t, lr_tplus1)
            data_group_ids.append(group_id)
            data_predictor_keys.append(predictor_key)
            data_model_ids.append(model_id)
            data_pearsonrs.append(pearsonr)
            data_pearsonr_ps.append(pearsonr_p)

data_dict = {
    "dataset_name": dataset_name,
    "io_pearsonr": io_pearsonr,
    "io_pearsonr_p": io_pearsonr_p,
    "group_id": data_group_ids,
    "predictor_key": data_predictor_keys,
    "model_id": data_model_ids,
    "pearsonr": data_pearsonrs,
    "pearsonr_p": data_pearsonr_ps
}
df = pandas.DataFrame(data_dict)
df.to_csv(out_path)
print("Results saved at", out_path)

