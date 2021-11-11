import argparse
import data
import decoding_data
import measure
from measure import get_vector
import os
import pandas
from scipy import stats
import sklearn.linear_model
import training_data

parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str, help="name of test dataset")
parser.add_argument("group_id", type=str, help="trained model group id")
parser.add_argument("-o_dir", type=str,
    help="path to output directory to save data to (including the decoders")
args = parser.parse_args()
dataset_name = args.dataset_name
group_id = args.group_id
out_dir = args.o_dir
decoder_group_dir = out_dir

# Create directory to save output data to
try:
    os.mkdir(out_dir)
except FileExistsError:
    # Remove directory and all its contents and recreate it
    import shutil
    shutil.rmtree(out_dir)
    os.mkdir(out_dir)

# Define outcome variable vectors
inputs, targets, p_gens = data.load_test_data(dataset_name)
io_outputs = data.load_io_outputs(dataset_name)
gen_process = data.load_gen_process(dataset_name)
outcome_vector_dict = \
    measure.get_decoder_target_vector_dict_from_io_outputs_inputs(io_outputs, inputs, gen_process)

# Perform linear regression separately on each network
networks, network_ids = training_data.load_models_ids(group_id)
n_networks = len(networks)

df_network_id = []
df_predictor = []
df_outcome = []
df_decoder_fname = []
df_r2 = []
df_r = []
for i_network, network_id in enumerate(network_ids):
    # Define predictor variable vectors
    network = networks[i_network]
    a = network.forward_activations(inputs)
    predictor_vector_dict = measure.get_predictor_vector_dict_from_activations(a)
    # For each predictor, outcome pair:
    for predictor_key, predictor_vector in predictor_vector_dict.items():
        for outcome_key, outcome_vector in outcome_vector_dict.items():
            # print("linreg {:d}, {:} -> {:}".format(i_network, predictor_key, outcome_key))
            # Partition into training and validation sets for linear regression
            n_samples = outcome_vector.shape[0]
            n_validate = n_samples // 10
            n_train = n_samples - n_validate
            predictor_train = predictor_vector[0:n_train]
            predictor_test = predictor_vector[n_train:n_samples]
            outcome_train = outcome_vector[0:n_train]
            outcome_test = outcome_vector[n_train:n_samples]
            # fit linear regression
            decoder = sklearn.linear_model.LinearRegression(n_jobs=-1)
            decoder.fit(predictor_train, outcome_train)
            # test linear regression
            prediction_test = decoder.predict(predictor_test)
            r2 = sklearn.metrics.r2_score(outcome_test, prediction_test)
            # score = decoder.score(predictor_test, outcome_test)
            # assert abs(r2 - score) < 1e-6, "divergence in r2 calculations: {:} vs {:}".format(r2, score)
            r, _ =  stats.pearsonr(outcome_test[:, 0], prediction_test[:, 0])
            # assert abs(r2 - r * r) < 0.05, "divergence in r2 calculations: {:} vs {:}".format(r2, r * r)
            # save linear regression
            fname = decoding_data.get_decoder_fname(outcome_key, predictor_key, network_id)
            decoding_data.save_decoder(decoder_group_dir, decoder, fname)
            # append row to dataframe
            df_network_id.append(network_id)
            df_predictor.append(predictor_key)
            df_outcome.append(outcome_key)
            df_decoder_fname.append(fname)
            df_r2.append(r2)
            df_r.append(r)

df_dict = dict(
    dataset_name=dataset_name,
    group_id=group_id,
    network_id=df_network_id,
    predictor=df_predictor,
    outcome=df_outcome,
    decoder_fname=df_decoder_fname,
    r2=df_r2,
    r=df_r)
df = pandas.DataFrame(df_dict)

out_path = decoding_data.get_group_regression_data_path(decoder_group_dir)
df.to_csv(out_path)
print("data saved at", out_path)
# print("n_samples", n_samples)
# print("n_train", n_train)
# print("n_validate", n_validate)
