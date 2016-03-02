#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev
# License:  BSD 3 clause
"""
    Metric learning
"""
from __future__ import division, print_function

import sys
import random
import numpy as np
import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import EmpiricalCovariance, MinCovDet

logger = logging.getLogger(__name__)


def plot_manifold_learning(X, y, n_neighbors=10, n_components=2, colors=None,
                           figsize=(15, 8)):
    """ Comparison of Manifold Learning methods.

        Author: Jake Vanderplas -- <vanderplas@astro.washington.edu>
        Modified by Oleg Razgulyaev
    """
    from time import time

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import NullFormatter

    from sklearn import manifold

    # Next line to silence pyflakes. This import is needed.
    Axes3D

    if colors:
        color = np.array([colors[a] for a in y])
    else:
        color = y

    fig = plt.figure(figsize=figsize)

    plt.suptitle("Manifold Learning with %i points, %i neighbors"
                 % (X.shape[0], n_neighbors), fontsize=14)

    from sklearn.decomposition import PCA, KernelPCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    kpca = KernelPCA(n_components=2, kernel="rbf",
                     fit_inverse_transform=True, gamma=None)
    pca = PCA(n_components=2)
    transformer = make_pipeline(StandardScaler(), pca)
    X2 = transformer.fit_transform(X)
    transformer = make_pipeline(StandardScaler(), kpca)
    X3 = transformer.fit_transform(X)

    if False:
        try:
            # compatibility matplotlib < 1.0
            ax = fig.add_subplot(251, projection='3d')
            ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], c=color,
                       cmap=plt.cm.Spectral)
            ax.view_init(4, -72)
        except:
            ax = fig.add_subplot(251, projection='3d')
            plt.scatter(2[:, 0], X2[:, 2], c=color, cmap=plt.cm.Spectral)

        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.zaxis.set_major_formatter(NullFormatter())
    else:
        ax = fig.add_subplot(251)
        plt.scatter(X2[:, 0], X2[:, 1], c=color, cmap=plt.cm.Spectral)
        plt.title("PCA")
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')
        
        ax = fig.add_subplot(256)
        plt.scatter(X3[:, 0], X3[:, 1], c=color, cmap=plt.cm.Spectral)
        plt.title("KPCA")
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

    methods = ['standard', 'ltsa', 'hessian', 'modified']
    labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

    for i, method in enumerate(methods):
        t0 = time()
        try:
            Y = manifold.LocallyLinearEmbedding(
                    n_neighbors, n_components,
                    eigen_solver='auto',
                    method=method).fit_transform(X)
        except:
            Y = np.zeros(X.shape)
        t1 = time()
        print("%s: %.2g sec" % (methods[i], t1 - t0))

        ax = fig.add_subplot(252 + i)
        plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
        plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

    t0 = time()
    Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
    t1 = time()
    print("Isomap: %.2g sec" % (t1 - t0))
    ax = fig.add_subplot(257)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("Isomap (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    t0 = time()
    mds = manifold.MDS(n_components, max_iter=100, n_init=1)
    Y = mds.fit_transform(X)
    t1 = time()
    print("MDS: %.2g sec" % (t1 - t0))
    ax = fig.add_subplot(258)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("MDS (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    t0 = time()
    se = manifold.SpectralEmbedding(n_components=n_components,
                                    n_neighbors=n_neighbors)
    Y = se.fit_transform(X)
    t1 = time()
    print("SpectralEmbedding: %.2g sec" % (t1 - t0))
    ax = fig.add_subplot(259)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    Y = tsne.fit_transform(X)
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))
    ax = fig.add_subplot(2, 5, 10)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    plt.show()


class MahalanobisTransformer(BaseEstimator, TransformerMixin):
    """ Mahalanobis Transformer.
    """
    def fit(self, X, y, targets=None):
        if targets is None:
            self.targets_ = np.unique(y)
            self.targets_.sort()
        else:
            self.targets_ = targets

        Covariance = MinCovDet if False else EmpiricalCovariance
        self.covs_ = [Covariance().fit(X[y == target, :])
                      for target in self.targets_]
        return self

    def transform(self, X):
        X_new = [cov.mahalanobis(X)**0.5 for cov in self.covs_]
        return np.array(X_new).T


def test():
    print("tests ok")

if __name__ == '__main__':
    random.seed(1)
    import argparse
    parser = argparse.ArgumentParser(description='Tuner.')
    parser.add_argument('cmd', nargs='?', default='test')
    args = parser.parse_args()
    print(args, file=sys.stderr)

    if args.cmd == 'test':
        test()
    else:
        raise ValueError("bad cmd")
