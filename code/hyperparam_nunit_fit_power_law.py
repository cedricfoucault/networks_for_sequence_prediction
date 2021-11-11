import argparse
import numpy as np
import pandas
import sklearn.linear_model
import utils

parser = argparse.ArgumentParser()
parser.add_argument("data_path_fit")
parser.add_argument("data_path_extrapolation")
utils.add_arguments(parser, ["output"])
args = parser.parse_args()
data_path_fit = args.data_path_fit
data_path_extrapolation = args.data_path_extrapolation
out_path = args.output

n_units_extrapolation = 1000
n_units_fit_max = 45
n_units_fit_min = 2

df_fit = pandas.read_csv(data_path_fit)
df_extrapolation = pandas.read_csv(data_path_extrapolation)

# Get the data points to fit and extrapolate on
n_units_values_fit = np.sort(df_fit["n_units"].unique())
n_units_values_fit = n_units_values_fit[(
    (n_units_values_fit <= n_units_fit_max)
    & (n_units_values_fit >= n_units_fit_min))]
n_values_fit = len(n_units_values_fit)
performances_fit = np.empty((n_values_fit))
for i, n_units in enumerate(n_units_values_fit):
    performances_fit[i] = df_fit[df_fit["n_units"] == n_units]["mean_performance"].iloc[0]
performance_extrapolation_measured = df_extrapolation[df_extrapolation["n_units"] == n_units_extrapolation]["mean_performance"].iloc[0]

# We model the performance as a function of the number of units as the power law function:
# 100 - performance = c * (1 / N) ^ alpha
# By taking the logarithm, we obtain the linear model:
# log(100 - performance) = a + b log(N), where a = log(c) and b = -alpha
# We therefore fit the parameters a and b using a linear regression
# from the log number of units, log(N), to the log error, log(100 - performance).

log_nunits_fit = np.log(n_units_values_fit)
log_nunits_extrapolation = np.log(n_units_extrapolation)
log_errors_fit = np.log(100 - performances_fit)

linear_model = sklearn.linear_model.LinearRegression(fit_intercept=True)
linear_model.fit(log_nunits_fit[:, np.newaxis], log_errors_fit)
r2_fit = linear_model.score(log_nunits_fit[:, np.newaxis], log_errors_fit)

power_law_exponent_parameter = - linear_model.coef_[0]
power_law_scaling_parameter = np.exp(linear_model.intercept_)

log_error_extrapolation_predicted = linear_model.predict(np.array([[log_nunits_extrapolation]]))[0]
performance_extrapolation_predicted = 100 - np.exp(log_error_extrapolation_predicted)

# R2 on fitted data points
output_string = f"R2 of the linear regression on the fitted data points: {r2_fit}"
# Estimated parameters of the power law
output_string += f"\nEstimated exponent parameter of the power law: {power_law_exponent_parameter}"
output_string += f"\nEstimated scaling constant parameter of the power law: {power_law_scaling_parameter}"
# Predicted performance at the extrapolation data point
output_string += f"\nPerformance predicted by the power law at N={n_units_extrapolation} units: {performance_extrapolation_predicted}"
# Actual performance at the extrapolation data point
output_string += f"\nActual measured performance at N={n_units_extrapolation} units: {performance_extrapolation_measured}"

if out_path is not None:
    with open(out_path, 'w') as f_out:
        print(output_string, file=f_out)
    print("Results saved at", out_path)
else:
    print(output_string)
