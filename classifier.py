#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
    Classifiers
"""

import sys, random
import numpy as np
import logging

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn import metrics

import sklearn.linear_model as lm
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA

from base import check_n_jobs

logger = logging.getLogger(__name__)

class LassoCV_proba(lm.LassoCV):
  def predict_proba(self,X):
    print 'alpha_:',self.alpha_
    y = self.predict(X)
    y = 1./(1+np.exp(-(y-0.5)))
    return np.vstack((1-y,y)).T

class RidgeCV_proba(lm.RidgeCV):
  def predict_proba(self,X):
    logger.debug('alpha_=%s',self.alpha_)
    y = self.predict(X)
    if 0:
        y_min,y_max = y.min(),y.max()
        if y_max>y_min:
            y = (y-y_min)/(y_max-y_min)
    else:
        y = 1./(1+np.exp(-(y-0.5)))
    return np.vstack((1-y,y)).T

class KNeighborsClassifier_proba(KNeighborsClassifier):
  def predict_proba(X):
    y = super(KNeighborsClassifier_proba, self).predict_proba(X)
    y[np.isnan(y)]=0.5
    return y

class ConstClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, c = 0):
    self.c = c
  def fit(self, X, y=None):
    return self
  def predict_proba(self, X):
    X = np.asarray(X)
    y1=np.empty(X.shape[0]); 
    y1.fill(self.c)
    y_proba = np.vstack((1-y1,y1)).T
    return y_proba
  def predict(self, X):
    return self.predict_proba(X)[:,1]

class MeanClassifier(BaseEstimator, ClassifierMixin):
  def fit(self, X, y=None):
    return self
  def predict_proba(self, X):
    X = np.asarray(X)
    y1 = np.mean(X, axis=1)
    y_proba = np.vstack((1-y1,y1)).T
    return y_proba
  def predict(self, X):
    return self.predect_proba()[:,1]
        
class RoundClassifier(BaseEstimator, ClassifierMixin):
  """
    Classifier with rounding classes 
  """
  def __init__(self, est, rup=0, find_cutoff=False):
    self.est = est
    self.rup = rup
    self.find_cutoff = find_cutoff

  def fit(self, X, y):
    from imbalanced import find_best_cutoff,round_smote,round_down,round_up
    if self.rup > 0:
        X1,y1,_ = round_up(X,y) 
    elif self.rup < 0:
        if self.rup < -1:
            X1,y1 = round_smote(X,y) 
        else:
            X1,y1 = round_down(X,y) 
    else:
        X1,y1 = X,y
    self.est.fit(X1,y1)
    if self.find_cutoff:
        ypp = self.predict_proba(X)[:,1]
        self.cutoff = find_best_cutoff(y,ypp)
    else:
        self.cutoff = 0.5
    return self

  def predict_proba(self, X):
    X = np.asarray(X)
    if hasattr(self.est,'predict_proba'):
        y_proba = self.est.predict_proba(X)
    else:
        y = self.est.predict(X)
        y_proba = np.vstack((1-y,y)).T
    return y_proba
  def predict(self, X):
    ypp = self.predict_proba(X)[:,1]
    return  np.array(map(int,ypp>self.cutoff))


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

