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
