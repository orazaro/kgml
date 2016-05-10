#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev
# License:  BSD 3 clause
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cross_validation import cross_val_score


def calc_pca_scores(X, cv=None, to_scale=True, min_score=-1E4,
                    verbosity=0):
    """ Calculate PCA MLE scores and select optimal n_components.
    """
    def plot_scores(d_scores, optimal_d):
        n_components = np.arange(1, d_scores.size+1)
        plt.plot(n_components, d_scores, 'b', label='PCA scores')
        plt.xlim(n_components[0], n_components[-1])
        plt.xlabel('n components')
        plt.ylabel('cv scores')
        plt.legend(loc='lower right')
        plt.title('PCA MLE scores')
        plt.axvline(optimal_d, color='r')
        plt.show()

    if to_scale:
        X = StandardScaler().fit_transform(X)
    d_scores = []
    for n in range(1, X.shape[1]):
        pca = PCA(n_components=n)
        scores = cross_val_score(pca, X, cv=cv)
        scores = np.mean(scores)
        if scores < min_score:
            break
        d_scores.append(scores)
    d_scores = np.array(d_scores)
    optimal_d = np.argmax(d_scores)+1
    if verbosity > 0:
        plot_scores(d_scores, optimal_d)
    return optimal_d


def calc_fa_scores(X, cv=None, to_scale=True, min_score=-1E12,
                   verbosity=0):
    """ Calculate FA MLE scores and select optimal n_components.
    """
    def plot_scores(d_scores, optimal_d):
        n_components = np.arange(1, d_scores.size+1)
        plt.plot(n_components, d_scores, 'b', label='FA scores')
        plt.xlim(n_components[0], n_components[-1])
        plt.xlabel('n components')
        plt.ylabel('cv scores')
        plt.legend(loc='lower right')
        plt.title('FA MLE scores')
        plt.axvline(optimal_d, color='r')
        plt.show()

    if to_scale:
        X = StandardScaler().fit_transform(X)
    d_scores = []
    for n in range(1, X.shape[1]):
        pca = FactorAnalysis(n_components=n)
        scores = cross_val_score(pca, X, cv=cv)
        scores = np.mean(scores)
        # if scores < min_score: break
        d_scores.append(scores)
    d_scores = np.array(d_scores)
    optimal_d = np.argmax(d_scores)+1
    if verbosity > 0:
        plot_scores(d_scores, optimal_d)
    return optimal_d
