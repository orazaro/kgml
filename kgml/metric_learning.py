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
from metric_learn import ITML, LSML, SDML

logger = logging.getLogger(__name__)


def plot_manifold_learning(X, y, n_neighbors=10, n_components=2, colors=None,
                           figsize=(15, 8)):
    """ Comparison of Manifold Learning methods.

        Author: Jake Vanderplas -- <vanderplas@astro.washington.edu>
        Modified by: Oleg Razgulyaev
    """
    from time import time

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import NullFormatter
    from sklearn.neighbors import KNeighborsClassifier

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
    from sklearn import metrics
    from sklearn.cross_validation import cross_val_predict

    def estimate(X, y):
        try:
            estimator = make_pipeline(
                KNeighborsClassifier(n_neighbors=n_neighbors,
                                     weights="uniform"))
            y_pred = cross_val_predict(estimator, X, y, cv=8, n_jobs=-1)
            f1_score = metrics.f1_score(y, y_pred, average='macro')
            logloss = metrics.log_loss(y, y_pred)
            print('logloss:', logloss)
        except:
            f1_score = 0.0
            logloss = 0.0
        return f1_score, logloss

    print("{:<20s}{:>12s}{:>12s}".format('transformer', 'time(Sec)',
          'f1_score'))

    t0 = time()
    f1_score = estimate(X, y)
    t1 = time()
    print("{:20s}{:12.2g}{:12.2f}".format('None', t1 - t0, f1_score))

    t0 = time()
    pca = PCA(n_components=2)
    transformer = make_pipeline(StandardScaler(), pca)
    X2 = transformer.fit_transform(X)
    f1_score_1 = estimate(X2, y)
    t1 = time()
    print("{:20s}{:12.2g}{:12.2f}".format('PCA', (t1 - t0), f1_score_1))

    t0 = time()
    kpca = KernelPCA(n_components=2, kernel="rbf",
                     fit_inverse_transform=True, gamma=None)
    transformer = make_pipeline(StandardScaler(), kpca)
    X3 = transformer.fit_transform(X)
    f1_score_2 = estimate(X3, y)
    t1 = time()
    print("{:20s}{:12.2g}{:12.2f}".format('KPCA', (t1 - t0), f1_score_2))

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
        plt.title("PCA ({:.2f})".format(f1_score_1))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

        ax = fig.add_subplot(256)
        plt.scatter(X3[:, 0], X3[:, 1], c=color, cmap=plt.cm.Spectral)
        plt.title("KPCA ({:.2f})".format(f1_score_2))
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
        # print("%s: %.2g sec" % (methods[i], t1 - t0))
        f1_score = estimate(Y, y)
        t1 = time()
        print("{:20s}{:12.2g}{:12.2f}".format(methods[i], t1 - t0, f1_score))

        ax = fig.add_subplot(252 + i)
        plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
        # plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
        plt.title("{:s} ({:.2f})".format(labels[i], f1_score))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

    t0 = time()
    Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
    f1_score = estimate(Y, y)
    t1 = time()
    print("{:20s}{:12.2g}{:12.2f}".format('Isomap', t1 - t0, f1_score))
    ax = fig.add_subplot(257)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("Isomap ({:.2f})".format(f1_score))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    t0 = time()
    mds = manifold.MDS(n_components, max_iter=100, n_init=1)
    Y = mds.fit_transform(X)
    f1_score = estimate(Y, y)
    t1 = time()
    print("{:20s}{:12.2g}{:12.2f}".format('MDS', t1 - t0, f1_score))
    ax = fig.add_subplot(258)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("MDS ({:.2f})".format(f1_score))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    t0 = time()
    se = manifold.SpectralEmbedding(n_components=n_components,
                                    n_neighbors=n_neighbors)
    Y = se.fit_transform(X)
    f1_score = estimate(Y, y)
    t1 = time()
    print("{:20s}{:12.2g}{:12.2f}".format('SpectralEmbedding', t1 - t0,
                                          f1_score))
    ax = fig.add_subplot(259)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("SpectralEmbedding ({:.2f})".format(f1_score))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    Y = tsne.fit_transform(X)
    f1_score = estimate(Y, y)
    t1 = time()
    print("{:20s}{:12.2g}{:12.2f}".format('t-SNE', t1 - t0, f1_score))
    ax = fig.add_subplot(2, 5, 10)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("t-SNE ({:.2f})".format(f1_score))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    plt.show()


def test_plot_manifold_learning():
    from sklearn.datasets import make_circles
    np.random.seed(0)
    X, y = make_circles(n_samples=400, factor=.3, noise=.05)
    plot_manifold_learning(X, y, n_neighbors=10, colors=None)


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


# ------ supervised metric learning ----------#

# Convert ITML, LSML, SDML into sklearn compatible classes

NUM_CONSTRAINTS = 60


class ITML_sk(ITML):
    def fit(self, X, y):
        num_constraints = NUM_CONSTRAINTS
        constraints = ITML.prepare_constraints(y, len(X), num_constraints)
        return super(ITML_sk, self).fit(X, constraints)


class LSML_sk(LSML):
    def fit(self, X, y):
        num_constraints = NUM_CONSTRAINTS
        constraints = LSML.prepare_constraints(y, num_constraints)
        return super(LSML_sk, self).fit(X, constraints)


class SDML_sk(SDML):
    def fit(self, X, y):
        num_constraints = NUM_CONSTRAINTS
        constraints = SDML.prepare_constraints(y, len(X), num_constraints)
        return super(SDML_sk, self).fit(X, constraints)


def test():
    test_plot_manifold_learning()
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
