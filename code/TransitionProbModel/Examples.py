#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script. Present a few example applications for the Markov Model toolbox.

To do next:
    - add a user-defined prior bias in the leak & hmm case

@author: Florent Meyniel
"""
# general
import matplotlib.pyplot as plt
import numpy as np

# specific to toolbox
from MarkovModel_Python import IdealObserver as IO
from MarkovModel_Python import GenerateSequence as sg

# %% Binary sequence and order 0 transition probabilities

# Generate sequence
L = int(1e2)
seq = sg.ProbToSequence_Nitem2_Order0(np.hstack((0.25*np.ones(L), 0.75*np.ones(L))))
seq = sg.ConvertSequence(seq)['seq']

# Compute observer with fixed decay and HMM observers
# The decay is exponential: a value of N means that each observation is discounted by a factor
# exp(-1/N) at every time step.
# In the HMM observer, p_c is the a priori probability of a change point on each trial
# For HMM, 'resol' is the number on bins used for numeric estimation of integrals on a grid.
options = {'Decay': 10, 'p_c': 1/200, 'resol': 20}
out_fixed = IO.IdealObserver(seq, 'fixed', order=0, options=options)
out_hmm = IO.IdealObserver(seq, 'hmm', order=0, options=options)

# PRIORS for fixed decay (or window)
# The prior is specified as the parameters of the corresponding beta/dirichlet distribution.
# If the prior is symmetric, simply use e.g. options['prior_weight'] = 10 (all parameters will be
# 10). The defaults is options['prior_weight'] = 1
# For a custom prior, use e.g. options['custom_prior'] = {(0,): 1, (1,): 5} to specify the parameter
# corresponding to each item (or transition)

# Plot result
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(out_fixed[(0,)]['mean'], label='p(1) mean')
plt.plot(out_fixed[(0,)]['SD'], linestyle='--', label='p(1) sd')
plt.legend(loc='best'), plt.ylim([0, 1])
plt.title('Exponential decay')

plt.subplot(3, 1, 2)
plt.imshow(out_hmm[(0,)]['dist'], origin='lower')
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.title('HMM -- full distribution')

plt.subplot(3, 1, 3)
plt.plot(out_hmm[(0,)]['mean'], label='p(1) mean')
plt.plot(out_hmm[(0,)]['SD'], linestyle='--', label='p(1) sd')
plt.legend(loc='best'), plt.ylim([0, 1])
plt.title('HMM -- moments')

# %% Binary sequence and order 1 coupled transition probabilities

# Generate sequence
L = int(1e2)
seq = sg.ProbToSequence_Nitem2_Order1(
        np.vstack((
                np.hstack((0.25*np.ones(L), 0.75*np.ones(L))),
                np.hstack((0.75*np.ones(L), 0.25*np.ones(L)))
                )))
seq = sg.ConvertSequence(seq)['seq']

# Compute Decay observer and HMM
options = {'Decay': 10, 'p_c': 1/200, 'resol': 20}
out_fixed = IO.IdealObserver(seq, 'fixed', order=1, options=options)
out_hmm = IO.IdealObserver(seq, 'hmm', order=1, options=options)


# Plot result
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(out_fixed[(0, 0,)]['mean'], label='p(0|0)')
plt.plot(out_fixed[(1, 0,)]['mean'], label='p(0|1)')
plt.legend(loc='best')
plt.ylim([0, 1])

plt.subplot(3, 1, 2)
vmin, vmax = 0, np.max([np.max(out_hmm[(0, 0)]['dist']), np.max(out_hmm[(1, 0)]['dist'])])
plt.imshow(out_hmm[(0, 0)]['dist'], origin='lower', vmin=vmin, vmax=vmax)
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.ylabel('p(0|0)')

plt.subplot(3, 1, 3)
plt.imshow(out_hmm[(1, 0)]['dist'], origin='lower', vmin=vmin, vmax=vmax)
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.ylabel('p(0|1)')

# %% Sequence of 3 items and order 0 transition probabilities

# Generate sequence
L = int(1e2)
Prob = {0: np.hstack((0.10*np.ones(L), 0.50*np.ones(L))),
        1: np.hstack((0.10*np.ones(L), 0.20*np.ones(L))),
        2: np.hstack((0.80*np.ones(L), 0.30*np.ones(L)))}
seq = sg.ProbToSequence_Nitem3_Order0(Prob)
seq = sg.ConvertSequence(seq)['seq']

options = {'Decay': 10, 'p_c': 1/200, 'resol': 20}
out_fixed = IO.IdealObserver(seq, 'fixed', order=0, options=options)
out_hmm = IO.IdealObserver(seq, 'hmm', order=0, options=options)

# Plot result
plt.figure()
plt.subplot(4, 1, 1)
plt.plot(out_fixed[(0,)]['mean'], label='p(0)')
plt.plot(out_fixed[(1,)]['mean'], label='p(1)')
plt.plot(out_fixed[(2,)]['mean'], label='p(2)')
plt.legend(loc='best')
plt.ylim([0, 1])
plt.subplot(4, 1, 2)
vmin, vmax = 0, np.max([np.max(out_hmm[(0,)]['dist']),
                        np.max(out_hmm[(1,)]['dist']),
                        np.max(out_hmm[(2,)]['dist'])])
plt.imshow(out_hmm[(0,)]['dist'], origin='lower', vmin=vmin, vmax=vmax)
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.ylabel('p(0)')
plt.subplot(4, 1, 3)
plt.imshow(out_hmm[(1,)]['dist'], origin='lower', vmin=vmin, vmax=vmax)
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.ylabel('p(1)')
plt.subplot(4, 1, 4)
plt.imshow(out_hmm[(2,)]['dist'], origin='lower', vmin=vmin, vmax=vmax)
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.ylabel('p(2)')

# %% Binary sequence and order 1 transition probability: coupled vs. uncoupled

# Generate sequence
L = int(1e2)
seq = sg.ProbToSequence_Nitem2_Order1(
        np.vstack((
                np.hstack((0.75*np.ones(L), 0.25*np.ones(L))),
                np.hstack((0.25*np.ones(L), 0.75*np.ones(L)))
                )))
seq = sg.ConvertSequence(seq)['seq']

# Compute HMM observer for coupled (simply specify 'hmm') and uncoupled case (specified with
# 'hmm_uncoupled')
options = {'p_c': 1/200, 'resol': 20}
out_hmm = IO.IdealObserver(seq, 'hmm', order=1, options=options)
out_hmm_unc = IO.IdealObserver(seq, 'hmm_uncoupled', order=1, options=options)

# Plot result
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(out_hmm[(0, 0)]['mean'], 'g', label='p(0|0), coupled')
plt.plot(out_hmm[(1, 0)]['mean'], 'b', label='p(0|1), coupled')
plt.plot(out_hmm_unc[(0, 0)]['mean'], 'g--', label='p(0|0), unc.')
plt.plot(out_hmm_unc[(1, 0)]['mean'], 'b--', label='p(0|1), unc.')
plt.legend(loc='upper left')
plt.ylim([0, 1])
plt.title('Comparison of means')

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(-np.log(out_hmm[(0, 0)]['SD']), 'g', label='p(0|0), coupled')
plt.plot(-np.log(out_hmm[(1, 0)]['SD']), 'b', label='p(0|1), coupled')
plt.plot(-np.log(out_hmm_unc[(0, 0)]['SD']), 'g--', label='p(0|0), uncoupled')
plt.plot(-np.log(out_hmm_unc[(1, 0)]['SD']), 'b--', label='p(0|1), uncoupled')
plt.legend(loc='upper left')
plt.title('Comparison of confidence')

# %% Estimate volatility of a binary sequence with order 1 coupled transition probability

# Generate sequence
L = int(1e2)
seq = sg.ProbToSequence_Nitem2_Order1(
        np.vstack((
                np.hstack((0.75*np.ones(L), 0.25*np.ones(L))),
                np.hstack((0.25*np.ones(L), 0.75*np.ones(L)))
                )))
seq = sg.ConvertSequence(seq)['seq']

# Compute HMM, but assumes (rather than learn) a given volatility level. Specify 'hmm' for this.
options = {'resol': 20, 'p_c': 1/L}
out_hmm = IO.IdealObserver(seq, 'hmm', order=1, options=options)

# Compute the full HMM that learn volatility from the data. Specify 'hmm+full' for this.
grid_nu = 1/2 ** np.array([k/2 for k in range(20)])
options = {'resol': 20, 'grid_nu': grid_nu, 'prior_nu': np.ones(20)/20}
out_hmm_full = IO.IdealObserver(seq, 'hmm+full', order=1, options=options)

# Plot result
plt.figure()
plt.subplot(3, 1, 1)
plt.imshow(out_hmm_full['volatility'])
expo = np.log(grid_nu)/np.log(1/2)
plt.yticks(ticks=[0, len(expo)/2, len(expo)],
                  labels=[f"1/{2**expo[0]:.0f}",
                          f"1/{2**expo[int(len(expo)/2)]:.0f}",
                          f"1/{2**expo[-1]:.0f}"])
plt.title('Volatility estimate')

plt.subplot(3, 1, 2)
plt.imshow(out_hmm_full[(0, 0)]['dist'], origin='lower')
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.title('p(0|0), full inference')

plt.subplot(3, 1, 3)
plt.imshow(out_hmm[(0, 0)]['dist'], origin='lower')
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.title('p(0|0), assuming vol.=1/L')

# %% Estimate volatility of a binary sequence with order 1 uncoupled transition probability

# Generate sequence
L = int(1e2)
seq = sg.ProbToSequence_Nitem2_Order1(
        np.vstack((
                np.hstack((0.25*np.ones(L), 0.75*np.ones(L))),
                np.hstack((0.25*np.ones(L), 0.75*np.ones(L)))
                )))
seq = sg.ConvertSequence(seq)['seq']

# Compute the full HMM that learn volatility from the data, while assuming that transition
# probabilities have uncoupled change points. Specify 'hmm_uncoupled' for this.
grid_nu = 1/2 ** np.array([k/2 for k in range(20)])
options = {'resol': 20, 'grid_nu': grid_nu, 'prior_nu': np.ones(20)/20}
out_hmm_unc_full = IO.IdealObserver(seq, 'hmm_uncoupled+full', order=1, options=options)
out_hmm_full = IO.IdealObserver(seq, 'hmm+full', order=1, options=options)

# Compute the HMM that assumes a given volatility level, for the uncoupled case. 
options = {'resol': 20, 'p_c': 1/L}
out_hmm_unc = IO.IdealObserver(seq, 'hmm_uncoupled', order=1, options=options)

# Plot result
plt.figure()
plt.subplot(4, 1, 1)
plt.imshow(out_hmm_unc_full['volatility'])
expo = np.log(grid_nu)/np.log(1/2)
plt.yticks(ticks=[0, len(expo)/2, len(expo)],
                  labels=[f"1/{2**expo[0]:.0f}",
                          f"1/{2**expo[int(len(expo)/2)]:.0f}",
                          f"1/{2**expo[-1]:.0f}"])
plt.title('Volatility estimate (unc)', **{'fontname': 'Arial', 'size': '12'})
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.subplot(4, 1, 2)
plt.imshow(out_hmm_unc_full[(0, 0)]['dist'], origin='lower')
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.title('p(0|0), full inference (unc)', **{'fontname': 'Arial', 'size': '12'})
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.subplot(4, 1, 3)
plt.imshow(out_hmm_unc[(0, 0)]['dist'], origin='lower')
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.title('p(0|0), vol=1/L (unc)', **{'fontname': 'Arial', 'size': '12'})
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.subplot(4, 1, 4)
plt.imshow(out_hmm_full[(0, 0)]['dist'], origin='lower')
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.title('p(0|0), full inference (coupled)', **{'fontname': 'Arial', 'size': '12'})
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
