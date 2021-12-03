#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:10:29 2021

@author: rodrigo
"""
import numpy as np
import scipy.stats as st
from scipy.linalg import cho_factor, cho_solve

from matplotlib import pyplot as plt

import sklearn

def hat_matrix(X, include_bias=True):
    """
    Compute hat matrix for design matrix X.

    :param np.array X: design matrix of dimensions (n x d),
    where n is the number of observations and d is the number of
    features.
    :param bool include_bias: if True (default), then include a bias column,
    in design matrix X (i.e. a column of ones - acts as an
    intercept term in a linear model).
    """
    if include_bias:
        X = np.hstack([np.ones([len(X), 1]), X])

    A = np.matmul(X.T, X)

    LL = cho_factor(A)
    return np.matmul(X, cho_solve(LL, X.T))


def anova(t, y_base, y_model, nparam_base, nparam_models,
          err_y=None, model_names=None):
    """
    Perform simple ANOVA analysis.

    :param np.array t: label array (dimensions (nsamples, 1) or (nsamples,))
    :param np.array y_base: predictions from base model
    (dimensions (nsamples, 1) or (nsamples,))
    :param np.array y_model: predictions from new (more complex) models
    (dimensions (nmodels, nsamples)
    :param int nparam_base: number of parameters of base model
    :param list nparam_models: list with number of parameters of new models
    :param list model_names: list with names of new models (optional)
    """
    y_model = np.atleast_2d(y_model)

    print('{:<10}\tdof\t{:<20}\tdof\tF-stat\tp-value'.format('Model',
                                                             'diferencia'))
    print('{:<10}\t---\t{:<20}\t---\t------\t-------'.format('-----',
                                                             '----------'))

    print('{:<10}\tN-{:d}'.format('Base', nparam_base))

    if err_y is None:
        err_y = y_base * 0.0 + 1.0

    for i, [y, npar] in enumerate(zip(y_model, nparam_models)):
        # Compute squared sums
        # Wikipedia https://en.wikipedia.org/wiki/F-test#Regression_problems
        # also Apunte by M.E. Szretter
        screg = np.sum((t - y_base)**2 / err_y**2)  # base model
        screg -= np.sum((t - y)**2 / err_y**2)  # new model
        screg /= (npar - nparam_base)  # normalise by difference in params

        scres = np.sum((t - y)**2 / err_y**2) / (len(t) - npar)

        fratio = screg/scres

        # Define appropiate F distribution
        my_f = st.f(dfn=(npar - nparam_base), dfd=(len(t) - npar))
        pvalue = 1 - my_f.cdf(fratio)

        if model_names is not None:
            name = model_names[i]
        else:
            name = 'New_{:d}'.format(i+1)

        printdict = {'model': name,
                     'dif': name+' - Base',
                     'npar': npar,
                     'dpar': npar - nparam_base,
                     'fratio': fratio,
                     'pvalue': pvalue
                     }
        # Print line in table
        print('{model:<10}\tN-{npar:d}\t{dif:<20}\t{dpar:d} '
              '\t{fratio:.4f}\t{pvalue:.2e}'.format(**printdict))

    return