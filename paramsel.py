#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
#
# Parameters selection tools

from __future__ import division
import os
import numpy as np

from six import string_types

from sklearn import grid_search

def param_grids(pgrid_str):
    if pgrid_str == 'svm_rbf':
        C_range = 10.0 ** np.arange(-3, 4)
        gamma_range = 10.0 ** np.arange(-4, 3)
        pg = dict(gamma=gamma_range, C=C_range)
    elif pgrid_str == 'svm_poly':
        pg = {'C':[0.001,0.01,0.1,1.0,10,100],'gamma':[0.1,0.01,0.001,0.0001],
                                                                'coef0':[0,1]}
    else:
        raise ValueError("Unknown pgrid_str: {}".format(pgrid_str))
    return pg

def psel_grid_search(model, X, y, param_grid, scoring='roc_auc', cv=4, n_jobs=-1, verbosity=1):
    """ Parameters selection using grid search.
    """
    if isinstance(param_grid, string_types):
        param_grid = param_grids(param_grid)

    clf = grid_search.GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=verbosity-2)
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

def test_psel_grid_search():
    pass

def test():
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
