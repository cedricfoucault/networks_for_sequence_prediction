import argparse
import data
import decoding_data
import pandas
import io_model
import measure
import numpy as np
import scipy.stats as stats
import sklearn.linear_model
import sklearn.feature_selection
import torch
import training_data

def get_prediction_features(predictions):
    p_tensor = predictions
    psquared_tensor = predictions ** 2
    log_p_tensor = torch.log(predictions)
    log_1minusp_tensor = torch.log(1 - predictions)
    entropy_tensor = - (predictions * torch.log(predictions) + (1 - predictions) * torch.log(1 - predictions))
    features = [p_tensor, psquared_tensor, log_p_tensor, log_1minusp_tensor, entropy_tensor]
    return torch.cat(features, dim=2)

def get_decoded_confidences_torch(inputs, network, decoder, predictor_key):
    activations = network.forward_activations(inputs)
    predictor_tensor_dict = measure.get_predictor_tensor_dict_from_activations(activations)
    return measure.get_decoded_tensor(decoder, predictor_tensor_dict[predictor_key])

def get_linreg_residuals_and_pearsonr_from_confidence_and_prediction_features(confidences, prediction_features):
    linreg = sklearn.linear_model.LinearRegression(n_jobs=-1)
    linreg.fit(prediction_features, confidences)
    confidences_predicted_from_features = linreg.predict(prediction_features)
    confidence_prediction_features_pearsonr, _ = stats.pearsonr(confidences_predicted_from_features.squeeze(), confidences.squeeze())
    confidence_residuals = confidences - confidences_predicted_from_features
    return confidence_residuals.squeeze(), confidence_prediction_features_pearsonr

parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str, help="name of test dataset")
parser.add_argument("--group_ids", nargs="+", type=str, help="trained model group ids")
parser.add_argument("--decoder_group_dirs", nargs="+", type=str, help="path to decoders' group directory")
parser.add_argument("-o", "--output", help="path to output file")
parser.add_argument("--verbose", action="store_true", default=False)
args = parser.parse_args()
dataset_name = args.dataset_name
group_ids = args.group_ids
decoder_group_dirs = args.decoder_group_dirs
out_path = args.output
verbose = args.verbose

outcome_key = "io_confidence"
predictor_key = "hidden_state"

n_groups = len(group_ids)
assert n_groups == len(decoder_group_dirs), "number of group ids in decoders don't match"

inputs, _ , _ = data.load_test_data(dataset_name)
gen_process = data.load_gen_process(dataset_name)

# Calculate the residual confidence of the ideal observer
# after regressing out features of the ideal observer's prediction
io = io_model.IOModel(gen_process)
io_outputs = io.get_output_stats(inputs)
io_confidences_torch = measure.get_confidence_from_sds(io_outputs["sd"])[:-1, :, :]
io_predictions_torch = io_outputs["mean"][:-1, :, :]
io_confidences = measure.get_vector(io_confidences_torch)
io_prediction_features = measure.get_vector(get_prediction_features(io_predictions_torch))
io_residual_confidences, pearsonr_io_confidence_and_linreg_io_prediction_features = \
    get_linreg_residuals_and_pearsonr_from_confidence_and_prediction_features(io_confidences, io_prediction_features)
pearsonr_io_residual_confidence_and_io_confidence, _ = stats.pearsonr(
            io_residual_confidences.squeeze(), io_confidences.squeeze())
io_predictions = measure.get_vector(io_predictions_torch)
# Calculate the mutual information between prediction and confidence in the ideal observer
mi_io_confidence_and_io_prediction = sklearn.feature_selection.mutual_info_regression(io_predictions, io_confidences.squeeze())[0]

if verbose:
    print(f"""pearson correlation between the ideal observer's confidence predicted from
        its prediction features and the ideal observer's true confidence:
        {pearsonr_io_confidence_and_linreg_io_prediction_features}""")
    print(f"""pearson correlation between the residuals of the ideal observer's confidence
        from the linear regression of its prediction features and the ideal observer's true confidence:
        {pearsonr_io_confidence_and_linreg_io_prediction_features}""")
    print(f"""mutual information between the ideal observer's confidence the ideal observer's prediction:
        {mi_io_confidence_and_io_prediction}""")

# For each network, compute the residual of the decoded confidence
# after regressing out features of the network's prediction and of the ideal observer's prediction.
# Also, compute the mutual information between the network's decoded confidence and the network's prediction
# and the mutual information between the network's decoded confidence and the ideal observer's prediction
data_group_ids = []
data_model_ids = []
data_pearsonrs_network_confidence_and_linreg_all_prediction_features = []
data_pearsonrs_network_residual_confidence_and_io_confidence = []
data_pearsonrs_network_residual_confidence_and_io_residual_confidence = []
data_mis_network_confidence_and_io_prediction = []
data_mis_network_confidence_and_network_prediction = []
for i_group, group_id in enumerate(group_ids):
    decoder_group_dir = decoder_group_dirs[i_group]
    networks, network_ids = training_data.load_models_ids(group_id)
    decoders = decoding_data.load_decoders(decoder_group_dir, outcome_key, predictor_key, network_ids)
    for i_network, network in enumerate(networks):
        model_id = network_ids[i_network]
        decoder = decoders[i_network]
        network_confidences_torch = get_decoded_confidences_torch(inputs, network, decoder, predictor_key)
        network_confidences = measure.get_vector(network_confidences_torch)
        network_predictions_torch = network(inputs)[:-1, :, :].detach()
        network_prediction_features = measure.get_vector(get_prediction_features(network_predictions_torch))
        all_prediction_features = np.concatenate((io_prediction_features, network_prediction_features), axis=1)
        network_residual_confidences, pearsonr_network_confidence_and_linreg_all_prediction_features = \
            get_linreg_residuals_and_pearsonr_from_confidence_and_prediction_features(network_confidences, all_prediction_features)
        pearsonr_network_residual_confidence_and_io_confidence, _ = stats.pearsonr(
            network_residual_confidences, io_confidences.squeeze())
        pearsonr_network_residual_confidence_and_io_residual_confidence, _ = stats.pearsonr(
            network_residual_confidences, io_residual_confidences)

        network_predictions = measure.get_vector(network_predictions_torch)
        mi_network_confidence_and_io_prediction = sklearn.feature_selection.mutual_info_regression(
            io_predictions, network_confidences.squeeze())[0]
        mi_network_confidence_and_network_prediction = sklearn.feature_selection.mutual_info_regression(
            network_predictions, network_confidences.squeeze())[0]

        data_group_ids.append(group_id)
        data_model_ids.append(model_id)
        data_pearsonrs_network_confidence_and_linreg_all_prediction_features.append(pearsonr_network_confidence_and_linreg_all_prediction_features)
        data_pearsonrs_network_residual_confidence_and_io_confidence.append(pearsonr_network_residual_confidence_and_io_confidence)
        data_pearsonrs_network_residual_confidence_and_io_residual_confidence.append(pearsonr_network_residual_confidence_and_io_residual_confidence)
        data_mis_network_confidence_and_io_prediction.append(mi_network_confidence_and_io_prediction)
        data_mis_network_confidence_and_network_prediction.append(mi_network_confidence_and_network_prediction)

data_dict = {
    "dataset_name": dataset_name,
    "pearsonr_io_confidence_and_linreg_io_prediction_features": pearsonr_io_confidence_and_linreg_io_prediction_features,
    "pearsonr_io_residual_confidence_and_io_confidence": pearsonr_io_residual_confidence_and_io_confidence,
    "mutual_information_io_confidence_and_io_prediction": mi_io_confidence_and_io_prediction,
    "group_id": data_group_ids,
    "model_id": data_model_ids,
    "pearsonr_network_confidence_and_linreg_all_prediction_features": data_pearsonrs_network_confidence_and_linreg_all_prediction_features,
    "pearsonr_network_residual_confidence_and_io_confidence": data_pearsonrs_network_residual_confidence_and_io_confidence,
    "pearsonr_network_residual_confidence_and_io_residual_confidence": data_pearsonrs_network_residual_confidence_and_io_residual_confidence,
    "mutual_information_network_confidence_and_io_prediction": data_mis_network_confidence_and_io_prediction,
    "mutual_information_network_confidence_and_network_prediction": data_mis_network_confidence_and_network_prediction
}
df = pandas.DataFrame(data_dict)
df.to_csv(out_path)
print("Results saved at", out_path)

