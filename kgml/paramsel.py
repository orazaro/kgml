#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
#
# Parameters selection tools

from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as plt

from six import string_types

from sklearn import grid_search, cross_validation

def param_grids(pgrid_str):
    if pgrid_str == 'svm_rbf':
        C_range = 10.0 ** np.arange(-3, 4)
        gamma_range = 10.0 ** np.arange(-4, 3)
        pg = dict(gamma=gamma_range, C=C_range)
    elif pgrid_str == 'svm_poly':
        pg = {'C':[0.001,0.01,0.1,1.0,10,100],'gamma':[0.1,0.01,0.001,0.0001],
                                                                'coef0':[0,1]}
    elif pgrid_str == 'C':
        C_range = 10.0 ** np.arange(-4, 4)
        pg = dict(C=C_range)
    elif pgrid_str == 'gamma':
        gamma_range = 10.0 ** np.arange(-4, 3)
        pg = dict(gamma=gamma_range)
    else:
        raise ValueError("Unknown pgrid_str: {}".format(pgrid_str))
    return pg

def psel_grid_search(model, X, y, param_grid, scoring='roc_auc', cv=4, n_jobs=-1, verbosity=1,
        rs=None):
    """ Parameters selection using grid search.
    """
    if isinstance(param_grid, string_types):
        param_grid = param_grids(param_grid)

    if isinstance(cv, (float,)) and 0 < cv < 1:
        cv = cross_validation.StratifiedShuffleSplit(y, 1, test_size=cv, random_state=rs) 

    clf = grid_search.GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=int(verbosity>2))
    clf.fit(X, y)

    if verbosity > 0:
        name = model.name if hasattr(model,'name') else model.__class__.__name__
        print("Best parameters for the model {} found on development set:".format(name))
        print
        print("{}   Score: {}".format(clf.best_params_,clf.best_score_))

   
    if verbosity > 1:
        print
        print("Grid scores on development set:")
        print
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
    
    return clf.best_params_, clf.best_score_


def plot_lc_param_train_val(model, X_train, y_train, X_val, y_val, 
    pname, plist, ax=None, figsize=(14,6), verbosity=0, rs=None, y_min = 0.75):
    """ Plot learning curve for the features.
    """
    from sklearn import metrics
    s_train, s_val = [],[]
    if verbosity>0:
        print("{:>20s}{:>10s}{:>10s}".format(pname[:18],"train","val"))
        print("-"*40)
    for p in plist:
        setattr(model,pname,p)
        model.fit(X_train,y_train)

        y_train_proba = model.predict_proba(X_train)[:,1]
        s_train.append(metrics.roc_auc_score(y_train, y_train_proba))

        y_val_proba = model.predict_proba(X_val)[:,1]
        s_val.append(metrics.roc_auc_score(y_val, y_val_proba))
        if verbosity>0:
            print("{:20}{:10.3f}{:10.3f}".format(p,s_train[-1],s_val[-1]))

    if ax is None:
        fig,ax1 = plt.subplots(1,1,figsize=figsize)
    else:
        ax1 = ax
    ax1.set_title("Learning Curve")
    ax1.set_xlabel(pname)
    ax1.set_ylabel("Roc Auc Score")
    ax1.plot(plist, s_train,label='train')
    ax1.plot(plist, s_val,label='val')
    ymin, ymax = ax1.get_ylim()
    if ymin < y_min and ymax > y_min:
        ax1.set_ylim(y_min,ymax)
    plt.grid()
    plt.legend(loc='lower right')
    if ax is None: plt.show()

def plot_lc_param(model, X, y, pname, plist, test_size=0.20,
        ax=None, figsize=(14,6), verbosity=0, rs=None, y_min = 0.75):
    """ Plot learning curve for the features.
    """
    from sklearn import cross_validation

    X_train, X_val, y_train, y_val = cross_validation.train_test_split(
        X, y, test_size=test_size, random_state=rs)
    
    return plot_lc_param_train_val(model, X_train, y_train, X_val, y_val,
        pname, plist, ax=ax, figsize=figsize, verbosity=verbosity, rs=rs, y_min=y_min)

def plot_lc_param_cv(model, X, y, pname, plist, cv=3, n_jobs=1,
        ax=None, figsize=(14,6), verbosity=0, rs=None, y_min = 0.75):
    """ Plot learning curve for the features using CV.
    """
    from sklearn import metrics
    from model_selection import cross_val_predict_proba
    s_train, s_val = [],[]
    if verbosity>0:
        print("{:>20s}{:>10s}{:>10s}".format(pname[:18],"train","val"))
        print("-"*40)
    for p in plist:
        setattr(model,pname,p)
        X_train,y_train = X,y
        
        model.fit(X_train,y_train)
        y_train_proba = model.predict_proba(X_train)[:,1]
        s_train.append(metrics.roc_auc_score(y_train, y_train_proba))

        X_val,y_val = X,y
        y_val_proba,_ = cross_val_predict_proba(model, X_val, y_val, scoring=None, cv=cv,
                n_jobs=n_jobs, verbose=0, fit_params=None, pre_dispatch='2*n_jobs')

        s_val.append(metrics.roc_auc_score(y_val, y_val_proba))
        if verbosity>0:
            print("{:20}{:10.3f}{:10.3f}".format(p,s_train[-1],s_val[-1]))

    if ax is None:
        fig,ax1 = plt.subplots(1,1,figsize=figsize)
    else:
        ax1 = ax
    ax1.set_title("Learning Curve")
    ax1.set_xlabel(pname)
    ax1.set_ylabel("Roc Auc Score")
    ax1.plot(plist, s_train,label='train')
    ax1.plot(plist, s_val,label='val')
    ymin, ymax = ax1.get_ylim()
    if ymin < y_min and ymax > y_min:
        ax1.set_ylim(y_min,ymax)
    plt.grid()
    plt.legend(loc='lower right')
    if ax is None: plt.show()

#--- helpers -----------------#

def make_skewed_data(n_samples=5000,n_features=20,n_classes=2):
    from sklearn.datasets.samples_generator import (make_classification, make_regression)
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=2,
        n_clusters_per_class=2, n_informative=8, n_redundant=2,
        random_state=1)
    # create unbalamced classes
    plus = np.where(y>0)[0]
    minus = np.where(y<=0)[0]
    plus_sel = random.sample(plus,int(len(plus)/25))
    sel = np.r_[minus,plus_sel]
    np.sort(sel)
    return X[sel,:],y[sel]

#--- tests -----------------#


def test_psel_grid_search():
    import sklearn.linear_model as lm
    X, y = make_skewed_data(n_samples=5000)
    print np.unique(y, return_counts=True)
    clf = lm.LogisticRegression(penalty='l2', dual=True, C=1.0,
            class_weight='auto', random_state=1,
                    solver='lbfgs')
    param_grid = param_grids('C')
    psel_grid_search(clf, X, y, param_grid, scoring='roc_auc', cv=4, n_jobs=-1, verbosity=2)
    psel_grid_search(clf, X, y, param_grid, scoring='roc_auc', cv=0.2, n_jobs=-1, verbosity=2)

def test():
    test_psel_grid_search()
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
