#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause

from __future__ import division
import os
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, mean_absolute_error, make_scorer
from sklearn.metrics import roc_curve, auc
from collections import defaultdict


def precision_sensitivity_specificity(y_true, y_proba, threshold=0.5):
    pr = np.asarray(y_proba > threshold, dtype=int)
    tr = np.asarray(y_true, dtype=int)
    tr_neg = (tr+1)%2
    pr_neg = (pr+1)%2
    tp = np.sum(pr & tr)
    tn = np.sum(pr_neg & tr_neg)
    fp = np.sum(pr & tr_neg)
    fn = np.sum(pr_neg & tr)

    # precision
    if (tp + fp) != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0
    # sensitivity or recall
    if (tp + fn) != 0:
        sensitivity = tp / (tp + fn)
    else:
        sensitivity = 0.0
    # specificity
    if (fp + tn) != 0:
        specificity = tn / (fp + tn)
    else:
        specificity = 0.0
    return precision,sensitivity,specificity

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss


def calc_roc_auc(y_test,y_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def estimate_scores(scores, scoring, sampling=True, n_sample=None, verbose=1):
    n_cv = len(scores)
    scores_mean = np.mean(scores)
    if scoring == 'roc_auc': #flip score < 0.5
        scores_mean = scores_mean if scores_mean >= 0.5 else 1 - scores_mean
    me = 1.96 * np.std(scores) 
    if sampling:
        if isinstance(scoring,basestring) and scoring=='accuracy':
            phat = scores_mean
            me = 1.96*np.sqrt(phat*(1-phat)/n_sample)
        else:
            me = me / np.sqrt(len(scores))

    if verbose > 0:
        print "%d Fold CV Score(%s): %.6f +- %.4f" % (n_cv, scoring, scores_mean, me,)
    return scores_mean, me

def bootstrap_632_gen(n, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    while True:
        train = np.random.randint(0,n,size=n)
        s_test = set(range(n)) - set(train)
        l_test = sorted(s_test)
        if len(l_test) > 0:
            if False:
                test_n = np.random.randint(0,len(l_test),size=n)
                test = np.asarray([l_test[i] for i in test_n])
            else:
                test = np.asarray(l_test)
            yield (train,test)

def bootstrap_632(n, n_iter, random_state=None):
    g = bootstrap_632_gen(n, random_state=random_state) 
    return [g.next() for i in range(n_iter)] 

def cv_select(y, random_state, n_cv, cv, test_size=0.1):
    if isinstance(cv,basestring):
        if cv == 'shuffle':
            return cross_validation.StratifiedShuffleSplit(y, n_cv, test_size=test_size, random_state=random_state)
        elif cv == 'loo':
            return cross_validation.LeaveOneOut(n=len(y))
        elif cv == 'kfold':
            return cross_validation.StratifiedKFold(y, n_folds=n_cv)
        elif cv == 'boot':
            return cross_validation.Bootstrap(len(y), n_iter=n_cv, train_size=(1-test_size), random_state=random_state)
        elif cv == 'boot632':
            return bootstrap_632(len(y), n_iter=n_cv, random_state=random_state)
        # for regression
        elif cv == '_shuffle':
            return cross_validation.ShuffleSplit(len(y), n_iter=n_cv, test_size=test_size, random_state=random_state)
        elif cv == '_kfold':
            return cross_validation.KFold(len(y), n_folds=n_cv)
        else:
            raise ValueError("bad cv:%s"%cv)
    else:
        return cv

def cv_run(rd, X, y, random_state, n_cv=16, n_iter=0, n_jobs=-1, scoring='accuracy', cv='shuffle', test_size=0.1, sampling=True):
    """ possible scorong: accuracy,roc_auc,precision,average_precision,f1
    """
    n_jobs = 1 if os.uname()[0]=='Darwin' else n_jobs
    if n_cv == 0:
      n_cv = len(y) 
    if n_iter > 0:
        p =[]
        for i in range(n_iter):
            cv1 = cv_select(y, random_state+i, n_cv, cv, test_size)
            scores = cross_validation.cross_val_score(rd, X, y, cv=cv1, 
                scoring=scoring, n_jobs=n_jobs, verbose=0)
            scores_mean,_ = estimate_scores(scores,scoring,
                sampling=True,n_sample=len(y),verbose=1)
            p.append(scores_mean)
        scores_mean,me = estimate_scores(p,scoring,
                sampling=False,n_sample=len(y),verbose=1)
        if scoring == 'accuracy':
            phat = scores_mean
            print "\tme_binom_est =",1.96*np.sqrt(phat*(1-phat)/len(y))
    else:
        cv1 = cv_select(y, random_state, n_cv, cv, test_size)
        if isinstance(scoring,basestring) and scoring=='rmse':
            scores = cross_validation.cross_val_score(rd, X, y, cv=cv1, 
                scoring='mean_squared_error',
                n_jobs=n_jobs, verbose=0)
            scores = [np.sqrt(np.abs(e)) for e in scores]
        elif isinstance(scoring,basestring) and scoring=='nrmse':
            """normalized rmse"""
            scores = cross_validation.cross_val_score(rd, X, y, cv=cv1, 
                scoring='mean_squared_error',
                n_jobs=n_jobs, verbose=0)
            y_std = np.std(y)
            scores = [np.sqrt(np.abs(e))/y_std for e in scores]
        elif isinstance(scoring,basestring) and scoring=='mae':
            mae = make_scorer(mean_absolute_error, greater_is_better=False)
            scores = cross_validation.cross_val_score(rd, X, y, cv=cv1, 
                scoring=mae,
                n_jobs=n_jobs, verbose=0)
            scores = np.abs(scores)
        else:
            scores = cross_validation.cross_val_score(rd, X, y, cv=cv1, scoring=scoring,
                n_jobs=n_jobs, verbose=0)
        scores_mean,me = estimate_scores(scores,scoring,sampling,n_sample=len(y))
    return scores_mean,me


###------- Split into train and test with nonoverlapping pids --###

def iter_to_ids(pid_iter):
    """ From list of pids to 
            dict i_pid -> list of indices
            dict pid -> i_pid
    """
    ids_voc = dict()
    ids = defaultdict(list)
    for index,pid in enumerate(pid_iter):
        if pid not in ids_voc:
            ids_voc[pid] = len(ids_voc)
        ids[ids_voc[pid]].append(index)
    return ids,ids_voc

def y_to_ids(y, ids):
    """ From list of flags 0/1 to flags 0/1 of groups grouped by ids 
    """
    y_ids = np.array([np.any(y[ids[i]]) for i in range(len(ids))],dtype=int)
    return y_ids

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
    scores_mean,me = estimate_scores(scores,scoring,sampling,n_sample=len(y))
    return scores_mean,me

def test_precision_sensitivity_specificity():
    n = 50
    tr = np.asarray(np.random.uniform(0,1,n) > 0.5,dtype=int)
    pr = np.asarray(np.random.uniform(0,1,n) > 0.5,dtype=int)
    print "tr:",tr[:5]
    print "pr:",pr[:5]
    precision,sensitivity,specificity = precision_sensitivity_specificity(tr,pr)
    print "precision:",precision
    print "sensitivity:",sensitivity
    print "specificity:",specificity
    from sklearn.metrics import precision_recall_fscore_support
    print precision_recall_fscore_support(tr,pr)

def test():
    #bootstrap_632(10, 5)
    test_precision_sensitivity_specificity()
    print "tests ok"

if __name__ == '__main__':
    import random,sys
    random.seed(1)
    import argparse
    parser = argparse.ArgumentParser(description='ModSel.')
    parser.add_argument('cmd', nargs='?', default='test')
    args = parser.parse_args()
    print >>sys.stderr,args 
  
    if args.cmd == 'test':
        test()
    else:
        raise ValueError("bad cmd")
