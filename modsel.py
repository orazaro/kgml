#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
import os
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from sklearn.ensemble import RandomForestClassifier
import sklearn.linear_model as lm
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)

from sklearn import grid_search

def get_clf(cl,n_jobs=1,random_state=0):
    """ Select clasifier by name
    """
    lm1 = {'C':[0.0001, 0.001, 0.01, 0.1, 0.3, 1, 3, 10]}
    C_range = 10.0 ** np.arange(-5, 3)
    C_range = np.hstack([C_range,[0.3,3]])
    lm2 = dict(C=C_range)
    est2 = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                             C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=random_state)
    est1 = lm.LogisticRegression(penalty='l1', dual=False, tol=0.0001, 
                             C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=random_state)

    svm1 = {'C':[0.001,0.01,0.1,1.0,10],'gamma':[0.1,0.01,0.001,0.0001]}
    #C_range = 10.0 ** np.arange(-2, 9)
    #gamma_range = 10.0 ** np.arange(-5, 4)
    C_range = 10.0 ** np.arange(-3, 4)
    gamma_range = 10.0 ** np.arange(-4, 3)
    svm2 = dict(gamma=gamma_range, C=C_range)
    est3 = svm.SVR(kernel='rbf',verbose=0)
    svm3 = {'C':[0.001,0.01,0.1,1.0,10],'gamma':[0.1,0.01,0.001,0.0001],'coef0':[0,1]}
    est4 = svm.SVR(kernel='poly',degree=3,verbose=0)
    knn1 = {'n_neighbors':2**np.arange(0, 8)}
    gb1 = {'max_depth':[1,2,4,8],'n_estimators':[10,20,40,80,160]}
    rf1 = {'max_depth':[2,4,8,16,24,32]}

    if cl=='rf':
        clf = RandomForestClassifier(n_estimators=200, max_depth=2,
                max_features='auto',
                n_jobs=n_jobs, random_state=random_state, verbose=0)
    elif cl=='rf2':
        clf1 = RandomForestClassifier(n_estimators=200, max_depth=2,
                max_features='auto',
                n_jobs=n_jobs, random_state=random_state, verbose=0)
        clf = grid_search.GridSearchCV(clf1, rf1, cv=4, n_jobs=n_jobs, verbose=0)
    elif cl=='lr2':
        clf = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                             C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=random_state)

    elif cl=='lr1':
        clf = lm.LogisticRegression(penalty='l1', dual=False, tol=0.0001, 
                             C=1.0, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=random_state)
    elif cl=='lr2g':
        clf = grid_search.GridSearchCV(est2, lm1, cv=4, n_jobs=n_jobs, verbose=0)

    elif cl=='lr1g':
        clf = grid_search.GridSearchCV(est1, lm1, cv=4, n_jobs=n_jobs, verbose=0)
    elif cl=='svmRg':
        clf = grid_search.GridSearchCV(est3, svm2, cv=4, n_jobs=n_jobs, verbose=0)
    elif cl=='svmP3':
        clf = grid_search.GridSearchCV(est4, svm3, cv=4, n_jobs=n_jobs, verbose=0)
    elif cl=='mnb':
        clf = MultinomialNB(alpha=1.0)
    elif cl=='gnb':
        clf = GaussianNB()
    elif cl=='knn':
        clf = grid_search.GridSearchCV(KNeighborsClassifier(), knn1, cv=4, n_jobs=n_jobs, verbose=0)
    elif cl=='lda':
        clf = LDA()
    elif cl=='qda':
        clf = QDA()
    elif cl=='gb':
        clf = grid_search.GridSearchCV(
            GradientBoostingClassifier(learning_rate=0.1,
                random_state=random_state,verbose=0,subsample=1.0), 
            gb1, cv=4, n_jobs=n_jobs, verbose=0)
    else:
        raise ValueError("bad cl:%s"%cl)

    return clf

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

def bootstrap_632(n, n_iter, random_state=None):
    while n_iter > 0:
        train = np.random.randint(0,n,size=n)
        s_test = set(range(n)) - set(train)
        l_test = sorted(s_test)
        if len(l_test) > 0:
            if False:
                test_n = np.random.randint(0,len(l_test),size=n)
                test = np.asarray([l_test[i] for i in test_n])
            else:
                test = np.asarray(l_test)
            #print train,test
            yield (train,test)
            n_iter -= 1

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
        elif cv == 'boot632':
            return bootstrap_632(len(y), n_iter=n_cv, random_state=random_state)
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
        scores = cross_validation.cross_val_score(rd, X, y, cv=cv1, scoring=scoring,
        n_jobs=n_jobs, verbose=0)
        scores_mean,me = estimate_scores(scores,scoring,sampling,n_sample=len(y))
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
    scores_mean,me = estimate_scores(scores,scoring,sampling,n_sample=len(y))
    return scores_mean,me


def test():
    bootstrap_632(10, 5)
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
