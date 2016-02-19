#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev
# License:  BSD 3 clause
"""
    Feature selectors
"""
from __future__ import division, print_function

import sys
import random
import os
import logging

import numpy as np

from sklearn.feature_selection import f_regression
from sklearn.datasets.samples_generator import (make_classification,
                                                make_regression)
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.base import clone
from sklearn.externals.joblib import Parallel, delayed

logger = logging.getLogger(__name__)


def remove_noninformative_columns(df):
    """ Remove noninformative columns:
        - with variance < 1E-15
    """
    variance = np.var(df, axis=0)
    return df.iloc[:, list(variance>1E-15)]

def add_quadratic_features(df, predictors, rm_noninform=False):
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

from predictive_analysis import df_xyf
from model_selection import cross_val_predict_proba
from modsel import estimate_scores
from sklearn import (metrics, cross_validation)

def forward_cv_inner_loop(model,df,selected,candidate,target,scoring,
        cv1=None,n_folds=8):
    selected_candidate = selected + [candidate]
    X,y,features = df_xyf(df, predictors=selected_candidate, target=target)
    if cv1 is None:
        cv1 = cross_validation.StratifiedKFold(y,n_folds)
    y_proba, scores = cross_val_predict_proba(model, X, y, scoring=scoring, 
        cv=cv1, n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs')
    scores_mean, me = estimate_scores(scores, scoring, sampling=False, 
        verbose=0)
    return (scores_mean, candidate)

def forward_cv(df, predictors, target, model, scoring = 'roc_auc', 
        n_folds=8, n_jobs=-1, start=[], selmax=None, verbosity=0):
    """ Forward selection using model.

    Parameters
    ----------

    Returns
    -------
    selected: list
        selected predictors
    
    Example
    -------
    References
    ----------
    """
    
    X,y,features = df_xyf(df,predictors=predictors,target=target)
    remaining = set([e for e in features if e not in start])
    selected = list(start)
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        pre_dispatch='2*n_jobs'
        parallel = Parallel(n_jobs=n_jobs, verbose=0,
                    pre_dispatch=pre_dispatch)
        scores_with_candidates = parallel(delayed(forward_cv_inner_loop)(
            clone(model),df,selected,candidate,target,scoring,
            cv1=None,n_folds=8)
            for candidate in remaining)
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
        else:
            break
        if verbosity > 0:
            print("{:.4f}".format(current_score), ' '.join(selected))
        if selmax is not None and len(selected) >= selmax: break
    return selected

def forward_selected(data, response, selmax=16, verbosity=0):
    """Linear model designed by forward selection.

    Parameters
    ----------
    data : pandas DataFrame 
        with all possible predictors and response
    response: str 
        name of response column in data

    Returns
    -------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    selected: list
        selected predictors
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
            print(current_score, selected)
        if len(selected) >= selmax: break
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
                print("F_select(%d):"%len(b),b[:90],"...",b[-10:], file=sys.stderr)
            else:
                print("F_select(%d):"%len(b),b[:maxf], file=sys.stderr)
        def out2():
            print("F_select(%d):" % len(b), file=sys.stderr)
            def pr(m1,m2):
                for i in range(m1,m2):
                    row = b[i]
                    print("%10s %10.2f %15g %10d" % (row[3],row[0],row[1],row[2]), file=sys.stderr)
            n = min(len(b),maxf)
            m = 90 if n > 100 else n
            pr(0,m)
            if n > 100:
                print("...", file=sys.stderr)
                pr(len(b)-10,len(b))
        if verbose > 1:
            out2()
        else:
            out()
    return np.asarray(idx_sel, dtype=int)


# --- plot learning curves for feature selection ----------------#

def calc_auc_cv(model, df, predictors, target, cv=8, n_jobs=-1):
    from .predictive_analysis import df_xyf
    from .model_selection import cross_val_predict_proba
    from sklearn.metrics import roc_auc_score

    X, y, features = df_xyf(df, predictors=predictors, target=target)
    y_proba, _ = cross_val_predict_proba(
        model, X, y, scoring=None, cv=cv, n_jobs=n_jobs, verbose=0,
        fit_params=None, pre_dispatch='2*n_jobs')
    return roc_auc_score(y, y_proba)


def plot_lc_features(model, df_train, df_val, predictors, target, df_test=None,
                     cv=8, n_jobs=-1, ax=None, y_min=0.75):
    """ Plot learning curve for the features.
    """
    import matplotlib.pyplot as plt
    s_train, s_val, s_test = [], [], []
    nf = len(predictors)
    for i in range(1, (nf + 1)):
        s_train.append(calc_auc_cv(model, df_train, predictors[:i], target,
                       cv=cv, n_jobs=n_jobs))
        s_val.append(calc_auc_cv(model, df_val, predictors[:i], target, cv=cv,
                                 n_jobs=n_jobs))
        if df_test is not None:
            s_test.append(calc_auc_cv(model, df_test, predictors[:i], target,
                                      cv=cv, n_jobs=n_jobs))
    if ax is None:
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
    else:
        ax1 = ax
    ax1.set_title("Feature Learning Curve")
    ax1.set_xlabel("Features selected")
    ax1.set_ylabel("Roc Auc Score")
    ax1.plot(range(1, nf + 1), s_train, label='train')
    ax1.plot(range(1, nf + 1), s_val, label='val')
    ymin, ymax = ax1.get_ylim()
    if ymin < y_min and ymax > y_min:
        ax1.set_ylim(y_min, ymax)
    if df_test is not None:
        ax1.plot(range(1, nf + 1), s_test, label='test')
    ax1.set_xticks(np.arange(0, nf + 1, 1))
    plt.grid()
    plt.legend(loc='lower right')
    if ax is None:
        plt.show()

# --- tests ----------------------------#

def make_skewed_dataframe(n_samples=5000, n_features=20):
    from sklearn.datasets.samples_generator import (make_classification, )
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_classes=2,
        n_clusters_per_class=2, n_informative=8, n_redundant=2,
        random_state=random_state)
    # create unbalamced classes
    plus = np.where(y > 0)[0]
    minus = np.where(y <= 0)[0]
    plus_sel = random.sample(plus, int(len(plus) / 25))
    sel = np.r_[minus, plus_sel]
    np.sort(sel)
    names = ["f{:02d}".format(i) for i in range(X.shape[1])]
    df = pd.DataFrame(np.c_[X[sel, :], y[sel]], columns=names + ['goal'])
    return df


def test_plot_lc_features(plton=False):
    from sklearn.cross_validation import StratifiedShuffleSplit
    from kgml.classifier import get_clf
    model = get_clf('lr2')
    df = make_skewed_dataframe(n_samples=1000)
    y = df['goal']
    sss = StratifiedShuffleSplit(y, 1, test_size=0.5, random_state=0)
    train, test = list(sss)[0]
    print(df.shape, max(train), max(test))
    if plton:
        plot_lc_features(
            model, df.iloc[train, :], df.iloc[test, :],
            predictors=df.columns, target='goal', cv=8, n_jobs=-1, ax=None)


def test_f_regression_select():
    print("==> a lot of features")
    X, y = make_regression(n_samples=20000, n_features=200, n_informative=150,
                             shuffle=False, random_state=0)
    idx_sel = f_regression_select(X, y, verbose=2)
    print("==> few ones")
    X, y = make_regression(n_samples=200, n_features=20, n_informative=5, noise=0.5,
                             shuffle=False, random_state=0)
    idx_sel = f_regression_select(X, y, verbose=1)
    print("tests ok")

if __name__ == '__main__':
    random.seed(1)
    import argparse
    parser = argparse.ArgumentParser(description='Feature selectors.')
    parser.add_argument('cmd', nargs='?', default='test')
    args = parser.parse_args()
    print(args, file=sys.stderr)
  
    if args.cmd == 'test':
        test_f_regression_select()
    else:
        raise ValueError("bad cmd")
