#!/usr/bin/env python
# Script to delete previously created dataset
import argparse
import data

try:
    __IPYTHON__
    # if running from iPython
    # we want to make sure the modules are up to date
    import importlib
    importlib.reload(data)
except NameError:
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="name for the dataset", metavar="dataset_name")
    args = parser.parse_args()
    fbasename = args.name

    data.delete_train_test_data(fbasename)
