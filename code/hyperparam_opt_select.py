import argparse
import numpy as np
import pandas
import utils

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="path to csv file")
parser.add_argument("--mode", type=str, default="mean", help="path to csv file")
utils.add_arguments(parser, ["output"])
args = parser.parse_args()
data_path = args.data_path
out_path = args.output
mode = args.mode

assert mode in ["mean", "min"], f"unknown mode: {mode}"

df = pandas.read_csv(data_path)
colname = "mean_loss" if mode == "mean" else "min_loss"
optimal_entry = df.loc[df[colname].idxmin()]
optimal_entry.to_csv(out_path)
print("Results saved at", out_path)
