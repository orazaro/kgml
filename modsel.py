#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
import os
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

def calc_roc_auc(y_test,y_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def estimate_scores(scores, scoring, sampling=True, verbose=1):
  n_cv = len(scores)
  me = 1.96 * np.std(scores) 
  if sampling:
      me = me / np.sqrt(len(scores))
  scores_mean = np.mean(scores)
  if scoring == 'roc_auc': #flip score < 0.5
      scores_mean = scores_mean if scores_mean >= 0.5 else 1 - scores_mean
  if verbose > 0:
    print "%d Fold CV Score(%s): %.6f +- %.4f" % (n_cv, scoring, scores_mean, me,)
  return scores_mean, me

def cv_select(y, random_state, n_cv, cv, test_size=0.1):
    if isinstance(cv,basestring):
        if cv == 'shuffle':
            return cross_validation.StratifiedShuffleSplit(y, n_cv, test_size=test_size, random_state=random_state)
        elif cv == 'loo':
            return cross_validation.LeaveOneOut(n_cv)
        elif cv == 'kfold':
            return cross_validation.StratifiedKFold(y, n_folds=n_cv)
        elif cv == 'boot':
            return cross_validation.Bootstrap(len(y), n_iter=n_cv, train_size=(1-test_size), random_state=random_state)
        else:
            raise ValueError("bad cv:%s"%cv)
    else:
        return cv

def cv_run(rd, X, y, random_state, n_cv=16, n_jobs=-1, scoring='accuracy', cv='shuffle', test_size=0.1, sampling=True):
    """ possible scorong: accuracy,roc_auc,precision,average_precision,f1
    """
    n_jobs = 1 if os.uname()[0]=='Darwin' else n_jobs
    if n_cv == 0:
      n_cv = len(y) 
    if isinstance(cv,basestring) and cv=='kfold' and scoring=='accuracy':
        p =[]
        for i in range(10):
            cv1 = cv_select(y, random_state+i, n_cv, cv, test_size)
            scores = cross_validation.cross_val_score(rd, X, y, cv=cv1, 
                scoring=scoring, n_jobs=n_jobs, verbose=0)
            scores_mean,_ = estimate_scores(scores,scoring,
                sampling=False,verbose=1)
            p.append(scores_mean)
        scores_mean,me = estimate_scores(scores,scoring,
                sampling=False,verbose=1)
        phat = scores_mean
        print "me_binom_est =",1.96*np.sqrt(phat*(1-phat)/len(y))
    else:
        cv1 = cv_select(y, random_state, n_cv, cv, test_size)
        scores = cross_validation.cross_val_score(rd, X, y, cv=cv1, scoring=scoring,
        n_jobs=n_jobs, verbose=0)
        scores_mean,me = estimate_scores(scores,scoring,sampling)
    return scores_mean,me

def cv_run_ids(rd, X, y, ids, random_state, n_cv = 16, n_jobs=-1, scoring='accuracy', cv='shuffle', test_size=0.1, sampling=True):
    """ possible scorong: accuracy,roc_auc,precision,average_precision,f1
    """
    n_jobs = 1 if os.uname()[0]=='Darwin' else n_jobs
    n_ids = len(ids)
    y_ids = [y[ids[i][0]] for i in range(n_ids)]
    #print "y_ids:",sum(y_ids),len(y_ids)
    if n_cv == 0:
      n_cv = len(y_ids) 
    cv_ids = cv_select(y_ids, random_state, n_cv, cv, test_size)
    cv1 = []
    for (a,b) in cv_ids:
      a1 = []
      for i in a:
          a1 = a1 + ids[i]
      b1 = []
      for i in b:
          b1 = b1 + ids[i]
      cv1.append( (np.array(a1),np.array(b1)) )
    scores = cross_validation.cross_val_score(rd, X, y, cv=cv1, scoring=scoring,
    n_jobs=n_jobs, verbose=0)
    #print scores
    scores_mean,me = estimate_scores(scores,scoring,sampling)
    return scores_mean,me
