#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hidden markov model inference using numeric integration.
The inference estimates volatility from the sequence itself.
If order>0 transition probabilities are estimated, the inference assumes that
they change at independent moments (uncoupled change points).

@author: Florent Meyniel
"""
from . import Inference_ChangePoint as io_theta
from . import Inference_UncoupledChangePoint as io_unc
import numpy as np


def forward_updating_uncoupled(seq, lik, Alpha0, p_c):
    """
    Compute inference of Dirichlet / Bernoulli parameter for a sequence with missing observations
    Returns the posterior distribution:
        p(\theta_t | y_{1:t}, nu)
    """

    # Start from the prior
    Alpha = Alpha0[:, np.newaxis]

    # Iteratively update
    for item, is_missing in zip(seq.data, seq.mask):
        if is_missing:
            # Update without observation
            Alpha = np.hstack((Alpha,
                               io_theta.turn_posterior_into_prediction(Alpha=Alpha[:, -1],
                                                                       p_c=p_c)))
        else:
            # Update with observation
            Alpha = np.hstack((Alpha,
                               io_theta.forward_updating([item], lik=lik, order=0,
                                                         p_c=p_c, Alpha0=Alpha[:, -1])))

    # Compute marginal distributions
    marg_Alpha = io_theta.marginal_Alpha(Alpha, lik)

    return marg_Alpha


def post_prob_vol_one_value(seq, conv_seq, lik, theta_prior, nu, nu_prior, order, Nitem):
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
        seq: Sequence of observation
        lik: the likelihood of observation (from Inference_ChangePoint.likelihood_table)
        theta_prior: the prior probability of \theta (from Inference_ChangePoint.init_Alpha)
        nu: the volatility
        nu_prior: the prior probability of nu
        order: order of transition considered
        Nitem: number of items in the sequence
    """

    # Get the posterior theta (with forward updating, given volatility \nu), independently for
    # the different transition types
    theta_post = {}
    for cond in conv_seq.keys():
        post_all_items = forward_updating_uncoupled(conv_seq[cond], lik,
                                                    theta_prior, nu)
        for item in post_all_items.keys():
            theta_post[cond+item] = post_all_items[item]

    # Compute probability of transition from theta_i to theta_{i+1}
    # theta_i varies across ROWS and theta_{i+1} varies across COLUMNS
    # NB: assume the same prior for all transition types
    T = np.zeros((len(theta_prior), len(theta_prior)), dtype=float)
    for k in range(len(theta_prior)):
        T[k, :] = theta_prior*nu
        T[k, k] = 1-nu
        T[k, :] = T[k, :] / sum(T[k, :])

    # Compute the sequence of predictions, as a function of theta_{i-1}
    seq_obs_lik_given_prior = np.zeros((len(theta_prior), len(seq)))
    for t in range(len(seq)):
        if t == 0:
            # Consider the prior probability (theta_{i-1}) of this observation
            seq_obs_lik_given_prior[:, t] = lik[(seq[0],)]
        else:
            # Update the prior probability (theta_{i-1}) assuming that a change can happen
            seq_obs_lik_given_prior[:, t] = (np.dot(T, lik[(seq[t],)]))

    # At the t-th observation, take the posterior from the previous time step
    # NB: use the posterior of the relevant statistics (conditioned on the previous item(s))
    previous_posterior = np.hstack([theta_prior[:, np.newaxis] for _ in range(order)])
    for t in range(order, len(seq)):
        previous_posterior = np.hstack((theta_prior[:, np.newaxis],
                                        theta_post[tuple(seq[t-order:t+1])][:, :-1]))

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
    lik, grid = io_theta.likelihood_table(Nitem=Nitem, resol=theta_resol, order=0)

    # Re-code the sequence for each type of transition
    conv_seq = io_unc.convert_to_order0(seq=seq, Nitem=Nitem, order=order)

    # Initialize \theta, the statistics of observations, using a flat prior
    theta_prior = io_theta.init_Alpha(Dir_grid=grid,
                                      order=0,
                                      Dirichlet_param=[1 for _ in range(Nitem)])

    # Compute the un-normalized, log probability of the volatility (\nu)
    # and cache the posterior theta for each volatility
    unnorm_log_post = []
    cache_theta_post = []
    cache_T = []
    for nu, nu_prior in zip(vol_grid, vol_prior):
        res = post_prob_vol_one_value(seq, conv_seq, lik, theta_prior, nu, nu_prior, order, Nitem)
        unnorm_log_post.append(res['unnorm_log_prob_vol'])
        cache_theta_post.append(res['theta_post'])
        cache_T.append(res['T'])

    # Normalize the posterior distribution of volatility (\nu)
    log_post_nu = np.array(unnorm_log_post)
    post_nu = np.exp(log_post_nu) / sum(np.exp(log_post_nu))

    return {'post_nu': post_nu,
            'theta_post_cond_nu': cache_theta_post,
            'T': cache_T}


def posterior_prediction(seq, order, Nitem, theta_resol, vol_grid, vol_prior):
    """
    Compute the posterior predictive distribution of observations, by also inferring volatility
    from the sequence of observations.
    """

    # Get the joint posterior probability of volatility (\nu)
    res = posterior_volatility(seq, order, Nitem, theta_resol, vol_grid, vol_prior)

    # Turn \theta into a posterior prediction of observations, marginalized over volatility
    # Do this for the posterior probability of each pattern
    theta_post_pred = {}
    for pattern in res['theta_post_cond_nu'][0].keys():
        theta_post_pred[pattern] = np.zeros(res['theta_post_cond_nu'][0][pattern].shape)
        for t in range(len(seq)):
            for post_nu, post_theta_given_nu, T_cond_nu in zip(res['post_nu'],
                                                               res['theta_post_cond_nu'],
                                                               res['T']):
                theta_post_pred[pattern][:, t] = \
                    theta_post_pred[pattern][:, t] + post_nu[t] * \
                    (np.dot(T_cond_nu, post_theta_given_nu[pattern][:, t]))

    return {'marg_theta': theta_post_pred, 'post_nu': res['post_nu']}
