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

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)


class MahalanobisClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """ Mahalanobis Classifier and Transformer.
    """
    def __init__(self, n_knn=5, est=None):
        self.n_knn = n_knn
        self.est = est

    def fit(self, X, y, targets=None):
        if targets is None:
            self.targets_ = np.unique(y)
            self.targets_.sort()
        else:
            self.targets_ = targets

        Covariance = MinCovDet if False else EmpiricalCovariance
        self.covs_ = [Covariance().fit(X[y == target, :])
                      for target in self.targets_]

        if self.est is None:
            self.est = KNeighborsClassifier(self.n_knn)

        return self

    def transform(self, X):
        X_new = []
        for cov in self.covs_:
            x_dist = cov.mahalanobis(X)**0.5
            print("x_dist:", x_dist)
            X_new.append(x_dist)
        X_new = np.array(X_new)
        print("X_new:", X_new.shape)
        return X_new.T

    def predict_proba(self, X):
        X = np.asarray(X)
        if hasattr(self.est, 'predict_proba'):
            y_proba = self.est.predict_proba(X)
        else:
            y = self.est.predict(X)
            y_proba = np.vstack((1-y, y)).T
        return y_proba

    def predict(self, X):
        if not self.find_cutoff and hasattr(self.est, 'predict'):
            return self.est.predict(X)
        ypp = self.predict_proba(X)[:, 1]
        return np.array(map(int, ypp > self.cutoff))


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
