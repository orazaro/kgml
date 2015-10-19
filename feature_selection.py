#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
    Feature selectors 
"""

import sys, random, os, logging

import numpy as np

from sklearn.feature_selection import f_regression
from sklearn.datasets.samples_generator import (make_classification,
                                                make_regression)
from sklearn.feature_selection import SelectPercentile, f_classif, chi2

logger = logging.getLogger(__name__)

def remove_noninformative_columns(df):
    """ Remove noninformative columns:
        - with variance < 1E-15
    """
    variance = np.var(df,axis=0)
    return df.iloc[:,list(variance>1E-15)]

def add_quadratic_features(df, predictors, rm_noninform=True):
    """ Add quadratic features based on the selected predictors
    
    Parameters
    ----------
    df: DataFrame
        dataset
    predictors: list of str
        column names to use to build the quadratic features

    Returns
    -------
    df1: DataFrame
        dataset with added features
    """
    import pandas as pd
    from itertools import combinations
    Xout = df.values
    columns = list(df.columns)
    X = df[predictors].values
    
    # add squares
    X1 = X * X
    Xout = np.c_[Xout,X1]
    columns.extend(["{}*{}".format(e,e) for e in predictors])
    #print columns
    
    # add combinations
    X2 = []
    for (i,j) in combinations(range(len(predictors)),2):
        X2.append(X[:,i]*X[:,j])
        columns.append("{}*{}".format(predictors[i],predictors[j]))
    X2 = np.vstack(X2).T
    Xout = np.c_[Xout,X2]
    #print columns
    df_out = pd.DataFrame(Xout,columns=columns)
    if rm_noninform:
        df_out = remove_noninformative_columns(df_out)
    return df_out 



def forward_selected(data, response, nmax=16, verbosity=0):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared

    Example:
    --------
    >> import pandas as pd
    >> url = "http://data.princeton.edu/wws509/datasets/salary.dat"
    >> data = pd.read_csv(url, sep='\\s+')
    >> model = forward_selected(data, 'sl')

    >> print model.model.formula
    >> # sl ~ rk + yr + 1

    >> print model.rsquared_adj
    >> # 0.835190760538
    
    References:
    -----------
    http://planspace.org/20150423-forward_selection_with_statsmodels/

    """
    import statsmodels.formula.api as smf
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
        if verbosity > 0:
            print current_score, selected
        nmax -= 1; if nmax <= 0: break
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model, selected


def f_regression_select(X, y, maxf = 300, pvals = True, names = None, verbose = 0, old_idx_sel=None):
    "Select features using f_regression"
    if names == None:
        names = ["f_%d"%(i+1) for i in range(X.shape[1])]
    if not old_idx_sel:
        old_idx_sel = range(X.shape[1])
    f=f_regression(X,y,center=False)
    # (F-value, p-value, col, name)
    a = [(f[0][i], f[1][i], old_idx_sel[i], names[i]) 
            for i in range(X.shape[1])]
    if pvals:
        a = [e for e in a if e[1]<0.05]
    a = sorted(a, reverse=True)
    idx_sel = [ e[2] for e in a[:maxf] ]
    if verbose > 0:
        b = a[:maxf]
        def out():
            if min(maxf,len(b)) > 100:
                print >>sys.stderr,"F_select(%d):"%len(b),b[:90],"...",b[-10:]
            else:
                print >>sys.stderr,"F_select(%d):"%len(b),b[:maxf]
        def out2():
            print >>sys.stderr,"F_select(%d):" % len(b)
            def pr(m1,m2):
                for i in range(m1,m2):
                    row = b[i]
                    print >>sys.stderr,"%10s %10.2f %15g %10d" % (row[3],row[0],row[1],row[2])
            n = min(len(b),maxf)
            m = 90 if n > 100 else n
            pr(0,m)
            if n > 100:
                print >>sys.stderr,"..."
                pr(len(b)-10,len(b))
        if verbose > 1:
            out2()
        else:
            out()
    return np.asarray(idx_sel, dtype=int)

def test_f_regression_select():
    print "==> a lot of features"
    X, y = make_regression(n_samples=20000, n_features=200, n_informative=150,
                             shuffle=False, random_state=0)
    idx_sel = f_regression_select(X, y, verbose=2)
    print "==> few ones"
    X, y = make_regression(n_samples=200, n_features=20, n_informative=5, noise=0.5,
                             shuffle=False, random_state=0)
    idx_sel = f_regression_select(X, y, verbose=1)
    print "tests ok"

if __name__ == '__main__':
    random.seed(1)
    import argparse
    parser = argparse.ArgumentParser(description='Feature selectors.')
    parser.add_argument('cmd', nargs='?', default='test')
    args = parser.parse_args()
    print >>sys.stderr,args 
  
    if args.cmd == 'test':
        test_f_regression_select()
    else:
        raise ValueError("bad cmd")
