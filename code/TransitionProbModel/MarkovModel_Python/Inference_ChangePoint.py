#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hidden markov model inference using numeric integration.
The inference assumes a given volatility level.
If order>0 transition probabilities are estimated, the inference assumes that
they change at the same moment (coupled change points).
In theory the code works for any number of items and any order of transition,
but in practice, it will crash due to memory limitation when order>1 and the
number of items > 3

To do next:
- change_marginalize currently corresponds to a flat prior
distribution. This could be improved (perhaps not to include biases in
transition probabilities, but at least biases in the base rate of occurence
of items)
- work on a sampling-based approximation when order>1 and number of items > 3

@author: Florent Meyniel
"""
import itertools
from operator import mul
from functools import reduce
from scipy.stats import dirichlet
import numpy as np


def likelihood_table(Nitem=2, resol=None, order=0):
    """
    Compute the likelihood of observations on a discretized parameter grid,
    for *Nitem* with *resol* values for each parameter, in the case of
    transitions with the specified *order*.

    If *order*=0, the observation likelihood is determined by Dirichlet
    parameters (or bernoulli parameters when *Nitem*=1); those parameters (and
    their combinations) are discretized using a grid. The number of dimensions
    of the grid is *Nitem*. The function outputs *Dir_grid*, which is a list
    of tuples, each tuple being a possible combination of dirichlet parameter
    values.

    If *order*>0, one must combine the possible Dirichlet parameters across
    transitions.

    The function outputs *observation_lik*, the likelihood of observations
    presented as a dictonary. The keys of this dictionary are the possible
    sequences of trailing and leading observations of interest given the
    specified order, and its values correspond to the discretized distribution
    of likelihoods into states. For instance, the key (0,1,2) corresponds to
    the sequence 0, then 1, then 2.

    """

    # Compute discretized parameter grid
    grid_param = np.linspace(0, 1, resol)

    # Get combinations of all possible (discretized) Dirichlet parameters
    # satisfying that their sum is 1.
    Dir_grid = [ntuple
                for ntuple in itertools.product(grid_param, repeat=Nitem)
                if np.isclose(sum(ntuple), 1)]

    # Compute likelihood of observation
    if order == 0:
        # we can compute the observation likelihood directly
        observation_lik = {}
        for item in range(Nitem):
            observation_lik[(item,)] = \
                np.array([combi[item] for combi in Dir_grid], dtype='float')

    else:
        # combine the values of Dirichlet parameters across higher-order
        # transitions

        # get likelihood when states are combined across patterns
        Dir_grid_combi = [combi for combi
                          in itertools.product(Dir_grid, repeat=Nitem**order)]

        # get list of trailing higher-order patterns
        pattern_list = [combi for combi
                        in itertools.product(range(Nitem), repeat=order)]

        # compute likelihood of current observation given the trailing pattern
        observation_lik = {}
        for index, pattern in enumerate(pattern_list):
            for obs in range(Nitem):
                observation_lik[pattern+(obs,)] = np.array(
                    [combi[index][obs] for combi in Dir_grid_combi],
                    dtype='float')

    return observation_lik, Dir_grid


def change_marginalize(curr_dist):
    """
    Compute the integral:
        \int p(\theta_t|y)p(\theta_{t+1}|\theta_t)) d\theta_t
        in which the transition matrix has zeros on the diagonal, and
        1/(n_state-1) elsewhere. In other words, it computes the updated
        distribution in the case of change point (but does not multiply by
        the prior probability of change point).

        NB: currently, the prior on transition is flat, and the prior on the
        base rate of occurence of item is also flat; we may want to change this
        latter aspect at least.
    """
    return (sum(curr_dist) - curr_dist) / (curr_dist.shape[0]-1)


def init_Alpha(Dir_grid=None, Dirichlet_param=None, order=None):
    """
    Initialize Alpha, which is the joint probability distribution of
    observations and parameter values.
    This initialization takes into account a bias in the dirchlet parameter
    (wit the constraint that the same bias applies to all transitions).
    Discretized state are sorted as in likelihood_table, such as the output
    of both functions can be combined.
    """
    # get discretized dirichlet distribution at quantitles' location
    dir_dist = [dirichlet.pdf(np.array(grid), Dirichlet_param)
                for grid in Dir_grid]

    # normalize to a probability distribution
    dir_dist = dir_dist / sum(dir_dist)

    # combine the values of those Dirichlet parameters across higher-order
    # transitions
    if order > 0:

        # get number of patterns used to condition the inference
        n_pattern = len(Dirichlet_param)**order

        # get joint likelihood when states are combined across patterns
        Alpha0 = [reduce(mul, combi) for combi
                  in itertools.product(dir_dist, repeat=n_pattern)]
    else:
        Alpha0 = dir_dist

    return np.array(Alpha0)


def forward_updating(seq=None, lik=None, order=None, p_c=None, Alpha0=None):
    """
    Update iteratively the Alpha, the joint probability of observations and parameters
    values, moving forward in the sequence.
    Alpha[t] is the estimate given previous observation, the t-th included.
    """

    # Initialize containers
    Alpha = np.ndarray((len(Alpha0), len(seq)))
    Alpha_no_change = np.ndarray((len(Alpha0), len(seq)))
    Alpha_change = np.ndarray((len(Alpha0), len(seq)))

    # Compute iteratively
    for t in range(len(seq)):
        if order > 0 and t < order:
            # simply repeat the prior
            Alpha_no_change[:, t] = (1-p_c)*Alpha0
            Alpha_change[:, t] = p_c*Alpha0
            Alpha[:, t] = Alpha0
        elif order == 0 and t == 0:
            # Update Alpha with the new observation
            Alpha_no_change[:, t] = (1-p_c) * lik[tuple(seq[t-order:t+1])] * Alpha0
            Alpha_change[:, t] = p_c * lik[tuple(seq[t-order:t+1])] * change_marginalize(Alpha0)
            Alpha[:, t] = Alpha_no_change[:, t] + Alpha_change[:, t]

            # Normalize
            cst = sum(Alpha[:, t])
            Alpha_no_change[:, t] = Alpha_no_change[:, t]/cst
            Alpha_change[:, t] = Alpha_change[:, t]/cst
            Alpha[:, t] = Alpha[:, t]/cst
        else:
            # Update Alpha with the new observation
            Alpha_no_change[:, t] = (1-p_c) * lik[tuple(seq[t-order:t+1])] * Alpha[:, t-1]
            Alpha_change[:, t] = p_c * lik[tuple(seq[t-order:t+1])] *\
                change_marginalize(Alpha[:, t-1])
            Alpha[:, t] = Alpha_no_change[:, t] + Alpha_change[:, t]

            # Normalize
            cst = sum(Alpha[:, t])
            Alpha_no_change[:, t] = Alpha_no_change[:, t]/cst
            Alpha_change[:, t] = Alpha_change[:, t]/cst
            Alpha[:, t] = Alpha[:, t]/cst

    return Alpha


def turn_posterior_into_prediction(Alpha=None, p_c=None):
    """
    Turn the posterior into a prediction, taking into account the possibility
    of a change point
    """
    # Check dimensions
    if len(Alpha.shape) == 1:
        Alpha = Alpha[:, np.newaxis]

    # Initialize containers
    pred_Alpha = np.ndarray(Alpha.shape)

    # Update
    for t in range(Alpha.shape[1]):
        # Update Alpha, without a new observation but taking into account
        # the possibility of a change point
        pred_Alpha[:, t] = (1-p_c) * Alpha[:, t] + \
                           p_c * change_marginalize(Alpha[:, t])

    return pred_Alpha


def marginal_Alpha(Alpha, lik):
    """
    Compute the marginal distributions for all Dirichlet parameters and
    transition types
    """
    marg_dist = {}
    for pattern in lik.keys():
        # get grid of values
        grid_val = np.unique(lik[pattern])

        # initialize container
        marg_dist[pattern] = np.zeros((len(grid_val), Alpha.shape[1]))

        # marginalize over the dimension corresponding to the other patterns
        for k, value in enumerate(grid_val):
            marg_dist[pattern][k, :] = np.sum(Alpha[(lik[pattern] == value), :], axis=0)

    return marg_dist


def compute_inference(seq=None, resol=None, order=None, Nitem=None, p_c=None):
    """
    Wrapper function that computes the posterior marginal distribution, starting
    from a sequence
    """

    lik, grid = likelihood_table(Nitem=Nitem, resol=resol, order=order)
    Alpha0 = init_Alpha(Dir_grid=grid, order=order,
                        Dirichlet_param=[1 for k in range(Nitem)])
    Alpha = forward_updating(seq=seq, lik=lik, order=order, p_c=p_c, Alpha0=Alpha0)
    pred_Alpha = turn_posterior_into_prediction(Alpha=Alpha, p_c=p_c)
    marg_Alpha = marginal_Alpha(pred_Alpha, lik)
    return marg_Alpha
