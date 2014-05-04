#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    estimators tuner
"""

import sys, random
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer

from base import check_n_jobs

"add link coef_ to feature_importances_"
for cla in [RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor]:
    if 'coef_' not in dir(cla):
        cla.coef_ = property(lambda self:self.feature_importances_)

class MaeRegressor(BaseEstimator, RegressorMixin):
  """estimator with MAE score"""
  def __init__(self, clf, clftype="", 
        kernel='rbf', degree=3, gamma=0.0, coef0=0.0, tol=0.001, C=1.0,
        alpha=1.0, loss='huber', epsilon=0.1,
        n_estimators=10, max_features='auto', max_depth=None, min_samples_leaf=1,
        subsample=1.0, oob_score=False,
        random_state=None,
        verbose=0,
        ):
    self.clf = clf
    self.clftype = clftype
    self.kernel = kernel
    self.degree = degree
    self.gamma = gamma
    self.coef0 =coef0
    self.tol= tol
    self.C = C
    self.alpha = alpha
    self.loss = loss
    self.epsilon = epsilon
    self.n_estimators = n_estimators
    self.max_features = max_features
    self.max_depth = max_depth
    self.min_samples_leaf = min_samples_leaf
    self.subsample = subsample
    self.oob_score = oob_score
    self.random_state = random_state
    self.verbose = verbose
  def fit(self, X, y):
    if self.clftype == 'svm':
        self.clf.set_params(C=self.C,gamma=self.gamma)
    elif self.clftype == 'lm':
        self.clf.set_params(alpha=self.alpha)
    elif self.clftype == 'sgd':
        self.clf.set_params(alpha=self.alpha,loss=self.loss,epsilon=self.epsilon,
            random_state=self.random_state)
    elif self.clftype == 'rf' or self.clftype == 'ef':
        self.clf.set_params(n_estimators=self.n_estimators, max_features=self.max_features, 
            max_depth=self.max_depth,min_samples_leaf=self.min_samples_leaf,
            oob_score = self.oob_score,
            random_state=self.random_state,
            verbose=self.verbose,n_jobs=check_n_jobs(-1))
    elif self.clftype == 'gb':
        self.clf.set_params(n_estimators=self.n_estimators, max_features=self.max_features, 
            max_depth=self.max_depth,min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
            verbose=self.verbose)
    self.clf.fit(X,y)
    return self
  def predict(self, X):
    return self.clf.predict(X)
  def score(self, X, y):
    y_pred = self.predict(X)
    return -mean_absolute_error(y, y_pred)
  @property
  def coef_(self):
    if 'coef_' in dir(self.clf):
        return self.clf.coef_
    elif 'feature_importances_' in dir(self.clf):
        return self.clf.feature_importances_
    else:
        raise ValueError('no coef_ or feature_importances_')


def scorer_gbr_lad(clf, X, y, verbose=1):
    """Scorer for GradientBoostingRegressor with los='lad' """
    y_pred = clf.predict(X)
    score = -mean_absolute_error(y, y_pred)
    if verbose >0:
        print >>sys.stderr,"Eout=",-score
        if 'staged_predict' in dir(clf):
            if verbose>0: print("Staged predicts (Eout)")
            for i,y_pred in enumerate(clf.staged_predict(X)):
                Eout = mean_absolute_error(y,y_pred)
                if verbose>0: print "tree %3d, test score %f" % (i+1,Eout)
    return score

"""
class GBRegressor(GradientBoostingRegressor):
  def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, min_samples_split=2, min_samples_leaf=1,
                 max_depth=3, init=None, random_state=None,
                 max_features=None, alpha=0.9, verbose=0):

    super(GBRegressor, self).__init__(loss, learning_rate, n_estimators,
                 subsample, min_samples_split, min_samples_leaf,
                 max_depth, init, random_state,
                 max_features, alpha, verbose)
"""



def test():
    print "tests ok"

if __name__ == '__main__':
    random.seed(1)
    import argparse
    parser = argparse.ArgumentParser(description='Tuner.')
    parser.add_argument('cmd', nargs='?', default='test')
    args = parser.parse_args()
    print >>sys.stderr,args 
  
    if args.cmd == 'test':
        test()
    else:
        raise ValueError("bad cmd")

