import argparse
import data
import decoding_data
import measure
import os
import pandas
from scipy import stats
import sequence
import sklearn.linear_model
import training_data

parser = argparse.ArgumentParser()
parser.add_argument("dataset_name_train", type=str,
    help="name of dataset to train confidence decoders on")
parser.add_argument("dataset_name_test", type=str, help="name of test dataset")
parser.add_argument("group_id", type=str, help="trained model group id")
parser.add_argument("-o", "--output", help="path to output file")
args = parser.parse_args()
dataset_name_train = args.dataset_name_train
dataset_name_test = args.dataset_name_test
group_id = args.group_id
out_path = args.output

# Define outcome variable vectors
train_inputs, train_targets, train_p_gens = data.load_test_data(dataset_name_train)
test_inputs, test_targets, test_p_gens = data.load_test_data(dataset_name_test)
train_io_outputs = data.load_io_outputs(dataset_name_train)
train_gen_process = data.load_gen_process(dataset_name_train)
test_io_outputs = data.load_io_outputs(dataset_name_test)
io_predictions_test = test_io_outputs["mean"]
test_gen_process = data.load_gen_process(dataset_name_test)
train_decoder_target_vector_dict = \
    measure.get_decoder_target_vector_dict_from_io_outputs_inputs(
        train_io_outputs, train_inputs, train_gen_process)
test_decoder_target_vector_dict = \
    measure.get_decoder_target_vector_dict_from_io_outputs_inputs(
        test_io_outputs, test_inputs, test_gen_process)
target_confidence_vector_train = train_decoder_target_vector_dict["io_confidence"]
target_confidence_vector_test = test_decoder_target_vector_dict["io_confidence"]
target_logodds_vector_train = train_decoder_target_vector_dict["io_evidence"]
target_logodds_vector_test = test_decoder_target_vector_dict["io_evidence"]
chance_loss_test = measure.get_chance_loss(test_inputs, test_targets)
bayes_loss_test = measure.get_loss(io_predictions_test, test_targets)
predictor_keys = decoding_data.CONFIDENCE_PREDICTOR_ALL_KEYS
predictor_key_logodds = "hidden_state"

networks, network_ids = training_data.load_models_ids(group_id)
n_networks = len(networks)

df_network_id = []
df_n_training_updates = []
df_predictor_key = []
df_prediction_performance = []
df_prediction_correlation = []
df_logodds_correlation = []
df_confidence_correlation = []
df_logodds_r2 = []
df_confidence_r2 = []
# Iterate over networks
for i_network, network_id in enumerate(network_ids):
    network = networks[i_network]
    parameters_through_training = \
        training_data.load_parameters_through_training_with_id(group_id, network_id)
    # Iterate network parameters through training
    for n_training_updates, parameters_dict in parameters_through_training.items():
        print("(i_network, n_training_updates)", (i_network, n_training_updates))
        # Restore network parameters for this training iteration
        network.restore_parameters_dict(parameters_dict)
        # Compute activations on dataset
        train_a = network.forward_activations(train_inputs)
        test_a = network.forward_activations(test_inputs)
        # Compute prediction accuracy as Pearson correlation on test set
        network_predictions_test = test_a["outputs"]
        prediction_correlation, _ = stats.pearsonr(
            sequence.get_numpy1D_from_tensor(io_predictions_test),
            sequence.get_numpy1D_from_tensor(network_predictions_test)
        )
        # Compute prediction performance
        network_loss_test = measure.get_loss(network_predictions_test, test_targets)
        prediction_performance = measure.get_percent_value(
            network_loss_test, chance_loss_test, bayes_loss_test)
        # Get predictor variable vectors for decoding
        predictor_vector_dict_train = \
            measure.get_predictor_vector_dict_from_activations(train_a)
        predictor_vector_dict_test = \
            measure.get_predictor_vector_dict_from_activations(test_a)
        # Compute decoded logodds correlation
        logodds_predictor_train = predictor_vector_dict_train[predictor_key_logodds]
        logodds_predictor_test = predictor_vector_dict_test[predictor_key_logodds]
        decoder = sklearn.linear_model.LinearRegression(n_jobs=-1)
        decoder.fit(logodds_predictor_train, target_logodds_vector_train)
        decoded_logodds_test = decoder.predict(logodds_predictor_test)
        logodds_correlation, _ =  stats.pearsonr(target_logodds_vector_test[:, 0],
            decoded_logodds_test[:, 0])
        logodds_r2 = sklearn.metrics.r2_score(target_logodds_vector_test,
                decoded_logodds_test)
        # Iterate over confidence decoder predictors
        for predictor_key in predictor_keys:
            if predictor_key not in predictor_vector_dict_train:
                continue
            predictor_train = predictor_vector_dict_train[predictor_key]
            predictor_test = predictor_vector_dict_test[predictor_key]
            # Fit linear regression decoder on confidence
            decoder = sklearn.linear_model.LinearRegression(n_jobs=-1)
            decoder.fit(predictor_train, target_confidence_vector_train)
            # Compute decoded confidence accuracy as Pearson correlation on test set
            decoded_confidence_test = decoder.predict(predictor_test)
            confidence_correlation, _ =  stats.pearsonr(target_confidence_vector_test[:, 0],
                decoded_confidence_test[:, 0])
            confidence_r2 = sklearn.metrics.r2_score(target_confidence_vector_test,
                decoded_confidence_test)
            # append row to dataframe
            df_network_id.append(network_id)
            df_n_training_updates.append(n_training_updates)
            df_predictor_key.append(predictor_key)
            df_prediction_performance.append(prediction_performance)
            df_prediction_correlation.append(prediction_correlation)
            df_logodds_correlation.append(logodds_correlation)
            df_confidence_correlation.append(confidence_correlation)
            df_logodds_r2.append(logodds_r2)
            df_confidence_r2.append(confidence_r2)
            # print("(prediction_correlation, logodds_correlation, confidence_correlation)",
            #     (prediction_correlation, logodds_correlation, confidence_correlation))


df_dict = dict(
    group_id=group_id,
    network_id=df_network_id,
    n_training_updates=df_n_training_updates,
    predictor_key=df_predictor_key,
    prediction_performance=df_prediction_performance,
    prediction_correlation=df_prediction_correlation,
    logodds_correlation=df_logodds_correlation,
    confidence_correlation=df_confidence_correlation,
    logodds_r2=df_logodds_r2,
    confidence_r2=df_confidence_r2
    )
df = pandas.DataFrame(df_dict)
df.to_csv(out_path)
print("data saved at", out_path)

