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

  def round_down(self,Xall,y1):
    p_zeros = [i for i,e in enumerate(y1) if e == 0]
    p_ones = [i for i,e in enumerate(y1) if e > 0]
    delta = len(p_zeros) - len(p_ones)
    if delta > 0:
        sel = random.sample(p_zeros,len(p_ones))
        sel = sel + p_ones
    elif delta < 0:
        sel = random.sample(p_ones,len(p_zeros))
        sel = sel + p_zeros
    else:
        return Xall,y1
    #print "round down:",len(p_zeros),len(p_ones),len(sel)
    return Xall[sel,:],y1[sel]

  def round_up(self,Xall,y1,ids=None):
    if not ids is None: 
        ids_inv = [None]*Xall.shape[0]
        for (k,v) in ids.iteritems():
            for i in v:
                ids_inv[i] = k
    p_zeros = [i for i,e in enumerate(y1) if e == 0]
    p_ones = [i for i,e in enumerate(y1) if e > 0]
    delta = len(p_zeros) - len(p_ones)
    if delta > 0:
        sel = [random.choice(p_ones) for _ in range(delta)]
    elif delta < 0:
        delta = -delta
        sel = [random.choice(p_zeros) for _ in range(delta)]
    else:
        return Xall,y1,ids
    X1 = [Xall]
    z1 = list(y1)
    j = Xall.shape[0]
    for i in sel:
        X1.append(Xall[i,:])
        z1.append(y1[i])
        if not ids is None:
            ids[ids_inv[i]].append(j)
        j += 1
    X1 = np.vstack(X1)
    z1 = np.array(z1).ravel()
    #print "round_up: 0:",len(p_zeros),"1:",len(p_ones),"X1:",X1.shape,"y1:",z1.shape,"j_last:",j
    return X1,z1,ids

  def find_best_cutoff(self,y1,ypp):
    from scipy import optimize
    def f(x,*params):
        y_true,ypp = params
        y_pred = np.array(map(int,ypp>x))
        res = metrics.f1_score(y_true, y_pred)
        #print "x:",x,"res:",res
        return -res
    rranges = (slice(0,1,0.01),)
    resbrute = optimize.brute(f, rranges, args=(y1,ypp), full_output=False,
                                  finish=optimize.fmin)
    print "resbrute:",resbrute
    return resbrute[0]

  def fit(self, X, y):
    if self.rup > 0:
        X1,y1,_ = self.round_up(X,y) 
    elif self.rup < 0:
        X1,y1 = self.round_down(X,y) 
    else:
        X1,y1 = X,y
    self.est.fit(X1,y1)
    if self.find_cutoff:
        ypp = self.predict_proba(X)[:,1]
        self.cutoff = self.find_best_cutoff(y,ypp)
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

