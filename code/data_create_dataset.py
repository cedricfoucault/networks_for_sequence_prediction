#!/usr/bin/env python
# Script to create train and test dataset. Also compute and save the Ideal Observer's estimates.
import argparse
import data
import generate as gen
import io_model

try:
    __IPYTHON__
    # if running from iPython
    # we want to make sure the modules are up to date
    import importlib
    importlib.reload(data)
    importlib.reload(gen)
    importlib.reload(io_model)
except NameError:
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="name for the dataset", metavar="dataset_name")
    parser.add_argument("--type", type=str, help="the type of the generative process", metavar="hm19", default="hm19")
    parser.add_argument("--pc", type=float, help="probability that the generative probability changes", metavar="p_change", default=None)
    parser.add_argument("--pclist", type=float, nargs='+', help="probability that the generative probability changes", metavar="p_change_list", default=None)
    parser.add_argument("--nmb", type=int, help="number of minibatches for training set", metavar="n_minibatches", default=None)
    parser.add_argument("--mbnseq", type=int, help="number of sequences by minibatches for training set", metavar="n_seq", default=None)
    parser.add_argument("--testnseq", type=int, help="number of sequences for test set", metavar="n_seq", default=None)
    parser.add_argument('--no-train', dest='train', action='store_false')
    parser.set_defaults(train=True)
    parser.add_argument('--no-test', dest='test', action='store_false')
    parser.set_defaults(test=True)
    args = parser.parse_args()
    fbasename = args.name
    p_type = args.type.lower()
    p_change = args.pc
    p_change_list = args.pclist
    n_minibatches = args.nmb
    minibatch_n_seq = args.mbnseq
    test_n_seq = args.testnseq
    train = args.train
    test = args.test
    verbose = True
    
    if p_type == 'hm19':
        if p_change != None:
            gen_process = gen.GenerativeProcessHeilbronMeyniel2019(p_change)
        else:
            gen_process = gen.GenerativeProcessHeilbronMeyniel2019()
    elif p_type == 'markov_coupled':
        assert p_change != None, "--pc must be provided for {:}".format(p_type)
        gen_process = gen.GenerativeProcessMarkovCoupled(p_change)
    elif p_type == 'markov_independent':
        assert p_change != None, "--pc must be provided for {:}".format(p_type)
        gen_process = gen.GenerativeProcessMarkovIndependent(p_change)
    elif p_type == 'bernoulli':
        assert p_change != None, "--pc must be provided for {:}".format(p_type)
        gen_process = gen.GenerativeProcessBernoulliRandom(p_change)
    elif p_type == 'bernoulli_multiple':
        assert p_change_list != None, "--pclist must be provided for {:}".format(p_type)
        gen_process = gen.GenerativeProcessBernoulliRandomMultiple(p_change_list)
    elif p_type == 'interleaved_altrep_markov_coupled':
        gen_process_1 = gen.GenerativeProcessAltRepCoupled(p_change)
        gen_process_2 = gen.GenerativeProcessMarkovCoupled(p_change)
    else:
        possible_types = ['interleaved_altrep_markov_coupled', 'hm19', 'markov_coupled', 'markov_independent' 'bernoulli', 'bernoulli_multiple']
        assert False, "--type should be one of {:}".format(possible_types)

    # Generate and save datasets
    is_interleaved = p_type.startswith("interleaved_")
    if train:
        argdict = {}
        if n_minibatches != None:
            argdict["train_n_minibatches"] = n_minibatches
        if minibatch_n_seq  != None:
            argdict["train_minibatch_n_sequences"] = minibatch_n_seq
        if is_interleaved:
            train_data = data.generate_interleaved_train_data(gen_process_1, gen_process_2, **argdict)
        else:
            train_data = data.generate_train_data(gen_process, **argdict)
    if test:
        argdict = {}
        if test_n_seq != None:
            argdict["test_n_sequences"] = test_n_seq
        if is_interleaved:
            test_data = data.generate_mixed_test_data(gen_process_1, gen_process_2, **argdict)
        else:
            test_data = data.generate_test_data(gen_process, **argdict)
    if is_interleaved:
        # Ugly, but we don't really care about the saved gen process in this case
        gen_process_saved = gen_process_2
    else:
        gen_process_saved = gen_process
    # Save the data
    if train:
        data.save_train_data(gen_process_saved, train_data, fbasename)
    else:
        data.save_gen_process(gen_process_saved, fbasename)
    if test:
        data.save_test_data(test_data, fbasename)
    


