#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hidden markov model inference using numeric integration.
The inference is made in two step:


To do next:
- change_marginalize currently corresponds to a flat prior
distribution. This could be improved (perhaps not to include biases in
transition probabilities, but at least biases in the base rate of occurence
of items)

@author: Florent Meyniel
"""

from . import Inference_ChangePoint as IO_hmm
import numpy as np
import itertools


def compute_inference(seq=None, resol=None, Nitem=None, p_c=None):
    """
    Compute inference of Dirichlet / Bernoulli parameter for a sequence with missing observations
    """
    order = 0
    lik, grid = IO_hmm.likelihood_table(Nitem=Nitem, resol=resol, order=order)
    Alpha0 = IO_hmm.init_Alpha(Dir_grid=grid, order=order,
                               Dirichlet_param=[1 for k in range(Nitem)])

    # Start from the prior
    Alpha = Alpha0[:, np.newaxis]

    # Iteratively update
    for item, is_missing in zip(seq.data, seq.mask):
        if is_missing:              # Update without observation
            Alpha = np.hstack((Alpha,
                               IO_hmm.turn_posterior_into_prediction(Alpha=Alpha[:, -1], p_c=p_c)))
        else:                       # Update with observation
            Alpha = np.hstack((Alpha,
                               IO_hmm.forward_updating([item], lik=lik, order=order,
                                                       p_c=p_c, Alpha0=Alpha[:, -1])))
    # Remove the prior from the sequence of posterior estimates
    Alpha = Alpha[:, 1:]

    # Compute posterior predictive distribution
    pred_Alpha = IO_hmm.turn_posterior_into_prediction(Alpha=Alpha, p_c=p_c)

    # Compute marginal distribution
    marg_Alpha = IO_hmm.marginal_Alpha(pred_Alpha, lik)

    return marg_Alpha


def convert_to_order0(seq=None, Nitem=None, order=None):
    """
    Detect items in the sequence that are preceded by the pattern "cond" (all patterns are
    searched for at the specified order). The converted sequence shows the detected item, and it
    masks all the other positions in the sequence.
    """
    converted_seq = {}
    for cond in itertools.product(range(Nitem), repeat=order):
        mask = np.array([True] * len(seq))
        cond_seq = np.zeros(len(seq), dtype=int)
        for item in range(Nitem):
            probe = list(cond) + [item]
            detected = np.array([False]*order +
                                [True if list(seq[k:k+order+1]) == probe else False
                                 for k in range(len(seq)-order)])
            cond_seq[detected] = item
            mask[detected] = False
        converted_seq[cond] = np.ma.masked_array(cond_seq, mask)
    return converted_seq
