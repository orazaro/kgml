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

from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


class MahalanobisClassifier(BaseEstimator, ClassifierMixin):
    """ Mahalanobis Classifier.
    """
    def __init__(self, est, rup=0, find_cutoff=False):
        self.est = est
        self.rup = rup
        self.find_cutoff = find_cutoff

    def fit(self, X, y):
        from imbalanced import (find_best_cutoff, round_smote,
                                round_down, round_up)
        if self.rup > 0:
            X1, y1, _ = round_up(X, y)
        elif self.rup < 0:
            if self.rup < -1:
                X1, y1 = round_smote(X, y)
            else:
                X1, y1, _ = round_down(X, y)
        else:
            X1, y1 = X, y
        self.est.fit(X1, y1)
        if self.find_cutoff:
            ypp = self.predict_proba(X)[:, 1]
            self.cutoff = find_best_cutoff(y, ypp)
        else:
            self.cutoff = 0.5
        return self

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
