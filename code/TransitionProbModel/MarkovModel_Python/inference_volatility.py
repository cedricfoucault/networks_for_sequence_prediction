#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hidden markov model inference using numeric integration.
The inference estimates volatility from the sequence itself.
If order>0 transition probabilities are estimated, the inference assumes that
they change at the same moment (coupled change points).
In theory the code works for any number of items and any order of transition,
but in practice, it will crash due to memory limitation when order>1 and the
number of items > 3

@author: Florent Meyniel
"""
from . import Inference_ChangePoint as io_theta
import numpy as np


def post_prob_vol_one_value(seq, lik, theta_prior, nu, nu_prior, order, Nitem):
    """
    Compute the (un-normalized) log probability of the volatility \nu:
         p(\nu|y_{1:t})
    as:
        p(\nu) * prod from i=1 to t {p(y_i|\nu, y_{1:i-1})}
    with:
        p(y_i|\nu, y_{1:i-1}) =
            int p(\theta_{i-1}|\nu, y{1:i-1}) * p(y_i|\theta_{i-1}, \nu)) d\theta_{i-1}
    Where \nu is the volatility and \theta is the probability of observation

    Parameters:
        seq: Sequence of observations
        lik: the likelihood of observations (from Inference_ChangePoint.likelihood_table)
        theta_prior: the prior probability of \theta (from Inference_ChangePoint.init_Alpha)
        nu: the volatility
        nu_prior: the prior probability of nu
        order: order of transitions considered
        Nitem: number of items in the sequence
    """

    # Get the posterior theta (with forward updating, given volatility \nu)
    theta_post = io_theta.forward_updating(seq=seq, lik=lik, order=order,
                                           p_c=nu, Alpha0=theta_prior)

    # Compute probability of transition from theta_i to theta_{i+1}
    # theta_i varies across ROWS and theta_{i+1} varies across COLUMNS
    T = np.zeros((len(theta_prior), len(theta_prior)), dtype=float)
    for k in range(len(theta_prior)):
        T[k, :] = theta_prior*nu
        T[k, k] = 1-nu
        T[k, :] = T[k, :] / sum(T[k, :])

    # Compute the sequence of predictions, as a function of theta_{i-1}
    seq_obs_lik_given_prior = np.zeros((len(theta_prior), len(seq)))
    for t in range(len(seq)):
        if order > 0 and t < order:
            # Consider a flat prior
            seq_obs_lik_given_prior[:, t] = 1/len(theta_prior)
        elif order == 0 and t == 0:
            # Consider the prior probability (theta_{i-1}) of this observation
            seq_obs_lik_given_prior[:, t] = lik[(seq[0],)]
        else:
            # Update the prior probability (theta_{i-1}) assuming that a change can happen
            seq_obs_lik_given_prior[:, t] = (np.dot(T, lik[tuple(seq[t-order:t+1])]))

    # At the t-th observation, take the posterior from the previous time step (i.e. the prediction)
    previous_posterior = np.hstack((theta_prior[:, np.newaxis], theta_post[:, :-1]))

    # Comptue the likelihood of each observation given the previous ones
    seq_lik_current_obs = [sum(prior * lik_given_prior)
                           for prior, lik_given_prior
                           in zip(previous_posterior.T, seq_obs_lik_given_prior.T)]

    return {'unnorm_log_prob_vol': np.log(nu_prior) + np.cumsum(np.log(seq_lik_current_obs)),
            'theta_post': theta_post,
            'T': T}


def posterior_volatility(seq, order, Nitem, theta_resol, vol_grid, vol_prior):
    """
    Compute the posterior probability of volatility, given the sequences received so far.
    Also cache the posterior distributions of \theta given each volatility level, the transition
    matrix given each volatility level.
    """

    # Get likelihood table for the forward inference of \theta, the statistics of observations.
    lik, grid = io_theta.likelihood_table(Nitem=Nitem, resol=theta_resol, order=order)

    # Initialize \theta, the statistics of observations (a flat prior)
    theta_prior = io_theta.init_Alpha(Dir_grid=grid,
                                      order=order,
                                      Dirichlet_param=[1 for _ in range(Nitem)])

    # Compute the un-normalized, log probability of the volatility (\nu)
    # and cache the posterior theta for each volatility
    unnorm_log_post = []
    cache_theta_post = []
    cache_T = []
    for nu, nu_prior in zip(vol_grid, vol_prior):
        res = post_prob_vol_one_value(seq, lik, theta_prior, nu, nu_prior, order, Nitem)
        unnorm_log_post.append(res['unnorm_log_prob_vol'])
        cache_theta_post.append(res['theta_post'])
        cache_T.append(res['T'])

    # Normalize the posterior distribution of volatility (\nu)
    log_post_nu = np.array(unnorm_log_post)
    post_nu = np.exp(log_post_nu) / sum(np.exp(log_post_nu))

    return {'post_nu': post_nu,
            'theta_post_cond_nu': cache_theta_post,
            'T': cache_T,
            'lik': lik}


def posterior_prediction(seq, order, Nitem, theta_resol, vol_grid, vol_prior):
    """
    Compute the posterior predictive distribution of observations, by also inferring volatility
    from the sequence of observations.
    """

    # Get the joint posterior probability of volatility (\nu) and statistics of observations
    # (\theta)
    res = posterior_volatility(seq, order, Nitem, theta_resol, vol_grid, vol_prior)

    # Turn \theta into a posterior prediction of observations, marginalized over volatility
    theta_post_pred = np.zeros(res['theta_post_cond_nu'][0].shape)
    for t in range(len(seq)):
        for post_nu, post_theta_given_nu, T_cond_nu in zip(res['post_nu'],
                                                           res['theta_post_cond_nu'],
                                                           res['T']):
            theta_post_pred[:, t] = \
                theta_post_pred[:, t] + post_nu[t] * \
                (np.dot(T_cond_nu, post_theta_given_nu[:, t]))

    # Get the marginal distribution corresponding to each transition of interest
    marg_theta = io_theta.marginal_Alpha(theta_post_pred, res['lik'])

    return {'marg_theta': marg_theta, 'post_nu': res['post_nu']}
