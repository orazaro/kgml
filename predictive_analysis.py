#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Predictive  analysis
"""
from __future__ import (division,)
import random, sys
import numpy as np
import pandas as pd
import warnings

random_state = 1

from sklearn import preprocessing, decomposition
from sklearn import svm, linear_model, ensemble, naive_bayes, neighbors
from sklearn import cross_validation, grid_search, metrics
from sklearn.feature_selection import RFECV, RFE

import matplotlib.pyplot as plt
from plot import plot_decision_line, plot_decision_boundary

def df_standardize(df, target):
    """ Convert DataFrame to standard form for ML: target at the last column
    
    Parameters
    ----------
    df: DataFrame
        dataset to strandartize as Pandas DataFrame
    target: str
        column name for the target

    Returns
    -------
    df: DataFrame
        dataset in the standard form for ML: target at the last column
    """
    if df.columns[-1] != target:
        new_columns = [s for s in df.columns if s != target]
        if len(new_columns) != len(df.columns) - 1:
            raise ValueError("target [{}] not in DataFrame column names!".format(target))
        new_columns.append(target)
        return df[new_columns]
    else:
        return df

def df_xyf(df, predictors=None, target=None, ydtype=None):
    """ Extract samples and target numpy arrays and its names from the DataFrame.

    Parameters
    ----------
    df: DataFrame shape=(n_samples, n_columns)
        dataset with all features and the targets
    predictors: list of str, optional (default=None)
        names of the predictors to use for training/predicting
        if None, than use all predictors except the target
    target: str, optional (default=None)
        target to predict
        if None, than use the last column in the DataFrame
    ydtype: dtype, optional (default=None)
        if not None, than convert y to this data type

    Returns
    -------
    X: array, shape=(n_samples, n_features)
        the train data samples with values of their features
    y: array, shape=(n_samples,))
        the targets
    feature_names: array-like of str
        feature names
    """
    if target is None:
        target = df.columns[-1]
    if predictors is None:
        predictors = np.array([e for e in df.columns if e != target])
    if ydtype is None:
        y = np.asarray(df[target].values)
    else:
        y = np.asarray(df[target].values,dtype=ydtype)
    return  (df.ix[:,predictors].values, 
            y,
            predictors)

def feature_selection_ET(df, predictors=None, target=None ,ax=None, isclass=True, 
        verbosity=0, nf=7, n_estimators=100, class_weight='auto',prank=False):
    """ Use ExtraTreesClassifier to calculate importances of predictors.

    Parameters
    ----------
    df: DataFrame shape=(n_samples, n_columns)
        dataset with all features and the targets
    predictors: list of str, optional (default=None)
        names of the predictors to use for training/predicting
        if None, than use all predictors except the target
    target: str, optional (default=None)
        target to predict
        if None, than use the last column in the DataFrame
    ax=None
    isclass=True
    verbosity=0
    nf=7
    n_estimators=100
    class_weight='auto'
    prank=False

    Returns
    -------
    """
    X, y, names = df_xyf(df, predictors=predictors, target=target)
    nf_all = X.shape[1]
    
    if verbosity > 1:
        print "names:", ",".join(names)
    if isclass:
        forest = ensemble.ExtraTreesClassifier(n_estimators=n_estimators,
                                      random_state=random_state,n_jobs=-1,class_weight=class_weight)
    else:
        forest = ensemble.ExtraTreesRegressor(n_estimators=n_estimators,
                                      random_state=random_state,n_jobs=-1)
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    if prank:
        # Print the feature ranking
        print("Feature ranking:")
        for i in range(nf_all):
            print("%2d. %25s (%10f)" % (i+1, names[indices[i]], importances[indices[i]]))

    if verbosity > 0:
        if ax is None:
            fig,ax1 = plt.subplots(1,1,figsize=(14,6))
        else:
            ax1 = ax
        # Plot the feature importances of the forest
        nf = nf_all if nf > nf_all else nf
        ax1.set_title("Feature importances")
        ax1.bar(range(nf), importances[indices][:nf],
               color="r", yerr=std[indices][:nf], align="center")
        anames = np.array(names)
        ax1.set_xlim([-1, nf])
        plt.xticks(range(nf), anames[indices][:nf],rotation='vertical')
        if ax is None: plt.show()
    
    names_sorted = [names[indices[i]] for i in range(len(predictors))]
    importances_sorted = [importances[indices[i]] for i in range(len(predictors))]
    return names_sorted,importances_sorted

#--- old stuff --------#

def get_clf(sclf,C=1.0,class_weight=None):
    if sclf == 'svm':
        clf = svm.SVC(kernel= "linear", C=C, class_weight=class_weight)
    elif sclf == 'svmr':
        clf = svm.SVC(kernel= "rbf", gamma=0.5, C=1, class_weight=class_weight)
    elif sclf == 'svmp':
        clf = svm.SVC(kernel= "poly", degree=3, gamma=0.1, C=5, class_weight=class_weight)
    elif sclf == 'lg1':
        clf = linear_model.LogisticRegression(penalty='l1', C=C, class_weight=class_weight)
    elif sclf == 'lg2':
        clf = linear_model.LogisticRegression(penalty='l2', C=C, class_weight=class_weight)
    elif sclf == 'lgCV':
        clf = linear_model.LogisticRegressionCV(penalty='l2', class_weight=class_weight)
    elif sclf == 'ridgeCV':
        clf = linear_model.RidgeCV()
    elif sclf == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        if not hasattr(RandomForestClassifier,'coef_'):
            RandomForestClassifier.coef_ = property(lambda self:self.feature_importances_)
        clf = RandomForestClassifier(n_estimators=100, max_depth=2, min_samples_leaf=2,
            class_weight=class_weight)
    elif sclf == 'gnb':
        clf = naive_bayes.GaussianNB()
    elif sclf == 'knc':
        clf = neighbors.KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError("bad sclf: {}".format(sclf))
    return clf


def do_analysis(fn,sclf,ax=None,sel=["PhaseTime","rdFar"],
    goal="Linebreak",pca=0,title="",
    toest=False,verbosity=0):
    X, y, names = data_prepare(fn, sel=sel, goal=goal, pca=pca, verbosity=verbosity-1)
  
    #class_weight = None
    class_weight = 'auto'
    
    sample_weight_constant = np.ones(len(X))
    
    weight_1 = np.sum(y<=0.5)/np.sum(y>0.5)
    sample_weight_1 = np.ones(len(X))
    sample_weight_1[y>0.5] = weight_1
    #print zip(y,sample_weight_1)

    if class_weight == 'auto':
        sample_weight = sample_weight_1
    else:
        sample_weight = sample_weight_constant 
   
    clf = get_clf(sclf,class_weight=class_weight)

    if toest==True:
        from kgml.modsel import cv_run
        n_cv=5
        res = cv_run(clf, X, y, random_state, n_cv=n_cv, scoring='accuracy')
        res = cv_run(clf, X, y, random_state, n_cv=n_cv, scoring='precision')
        res = cv_run(clf, X, y, random_state, n_cv=n_cv, scoring='recall')
        res = cv_run(clf, X, y, random_state, n_cv=n_cv, scoring='f1')
        #res = cv_run(clf, X, y, random_state, n_cv=n_cv, scoring='roc_auc')
    else: 
        if True:
            clf.fit(X, y)
        else:
            test_size = int(len(X)*0.5)
            clf.fit(X[:-test_size], y[:-test_size])
            correct_count = 0

            for x in range(1, test_size+1):
                if clf.predict(X[-x])[0] == y[-x]:
                    correct_count +=1

            if verbosity>0:
                print("Accuracy: ", (correct_count/test_size) * 100.00)
        if isinstance(fn,basestring):
            title = "file:{}  model:{}  weights:{}".format(fn[:10],sclf,class_weight)
        if ax is None:
            fig,ax1 = plt.subplots(1)
        else:
            ax1 = ax
        plot_decision_boundary(clf, X, y, ax=ax1, sample_weight=sample_weight, names=names, title=title) 
        if ax is None:
            plt.show()

def predict_evaluate_models(fn ,ax=None, sel=["PhaseTime","rdFar"], goal="Linebreak", 
    pca=0, verbosity=0, nfolds = 10, metrics_av="weighted", toRoundDown=True,
    sclfs=('svm','svmp','svmr','lgCV','gnb','rf','knc')):
    class_weight = 'auto'
    X, y, names = data_prepare(fn, sel=sel, goal=goal, pca=pca, verbosity=verbosity-1)
    if verbosity > 2:
        y_shuffled = y.copy()
        np.random.shuffle(y_shuffled)
        print "All zeros accuracy:",1.0-np.sum(y)/len(y) 
        print "y_shuffled f1_csore:",metrics.f1_score(y, y_shuffled)

    if nfolds >= len(y)/2:
        if verbosity > 0:
            print "cross_validation.LeaveOneOut"
        cv = cross_validation.LeaveOneOut(n=len(y))
    else:
        if verbosity > 0:
            print "cross_validation.StratifiedKFold"
        cv = cross_validation.StratifiedKFold(y, n_folds=nfolds)
    results = []
    for sclf in sclfs:
        clf = get_clf(sclf,class_weight=class_weight)
        y_pred = cross_validation.cross_val_predict(clf, X, y, cv=cv)
        #print "pred:",y_pred
        
        if toRoundDown and len(np.unique(y))==2:    # round down goal classes
            from kgml.imbalanced import round_down_sel
            sel = round_down_sel(y)
            if len(sel) > 0:
                y1,y1_pred = y[sel],y_pred[sel]
            else:
                y1,y1_pred = y,y_pred
        else:
            y1,y1_pred = y,y_pred
        
        res = [
            metrics.accuracy_score(y1, y1_pred),
            metrics.precision_score(y1, y1_pred, average=metrics_av),
            metrics.recall_score(y1, y1_pred, average=metrics_av),
            metrics.f1_score(y1, y1_pred, average=metrics_av),
            ]
        if verbosity > 0:
            print sclf,res 
        results.append( (sclf,res) )

    return results

def out_predictive_results(results):
    print "%10s%10s%10s%10s%10s" % ("classifier","accuracy","precision","recall","f1_score")
    for sclf,res in results:
        print "%10s%10.2f%10.2f%10.2f%10.2f" % tuple([sclf]+res)

def estimate_predictions(df, features, goal, sels = ('all','pca2'),
            nfolds=5, metrics_av="weighted"):
    for sel in sels:
        print "features:",sel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if sel[:3] == 'pca':
                pca = int(sel[3:])
            else:
                pca = 0
            results = predict_evaluate_models(fn=df,sel=features,goal=goal,pca=pca,
                nfolds=nfolds,metrics_av=metrics_av)
            out_predictive_results(results)
        print



def feature_selection_RFE(fn ,ax=None, sel="all", goal="Linebreak", isclass=True,
        verbosity=0, nf=7):
    X, y, names = data_prepare(fn, sel=sel, goal=goal, verbosity=verbosity-1)
    if verbosity > 1:
        print "names:", ",".join(names)
    
    # Create the RFE object and compute a cross-validated score.
    if isclass:
        #estimator = svm.SVC(kernel="linear",C=1.0)
        estimator = get_clf('svm')    
        scoring = 'f1'
        cv = cross_validation.StratifiedKFold(y, 2)
    else:
        if False:
            from sklearn.ensemble import RandomForestRegressor
            if not hasattr(RandomForestRegressor,'coef_'):
                RandomForestRegressor.coef_ = property(lambda self:self.feature_importances_)
            estimator = RandomForestRegressor(n_estimators=100, max_depth=2, min_samples_leaf=2)
        else:
            estimator = linear_model.RidgeCV()
        scoring = 'mean_squared_error'
        cv = 3

    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    if True:
        rfecv = RFECV(estimator=estimator, step=1, cv=cv, scoring=scoring)
    else:
        from kgml.rfecv import RFECVp
        f_estimator = get_clf('svm')
        rfecv = RFECVp(estimator=estimator,f_estimator=f_estimator, step=1, cv=cv, scoring=scoring)
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rfecv.fit(X, y)

    # Plot number of features VS. cross-validation scores
    ax.set_xlabel("Number of features selected")
    ax.set_ylabel("Cross validation score ({})".format(scoring))
    ax.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

    #print("Optimal number of features : %d" % rfecv.n_features_)
    best = names[rfecv.ranking_==1]

    rfe = RFE(estimator, n_features_to_select=1)
    rfe.fit(X,y)
    ranks = sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))

    # reorder best using ranks
    best_set = set(best)
    best = [name for (i,name) in ranks if name in best_set]
    #print "The best features:", ', '.join(best)
    assert len(best) == len(best_set)

    return best, ranks


def test(args):
    #predict_evaluate_models(args.fn,sel=args.sel,verbosity=2)
    if False:    
        fig,ax = plt.subplots()
        feature_selection_ET(args.fn ,ax=ax, sel="all", goal="Linebreak", verbosity=2)
        plt.show()
    if True:
        fig,ax = plt.subplots()
        feature_selection_RFE(args.fn ,ax=ax, sel="all", goal="Linebreak", verbosity=2)
        plt.show()
 
    print >>sys.stderr,"Test OK"
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Commands.')
    parser.add_argument('cmd', nargs='?', default='test',help="make_train|make_test")
    parser.add_argument('-rs', type=int, default=None,help="random_state")
    parser.add_argument('-clf', type=str, default='svm',help="classification model")
    parser.add_argument('-sel', type=str, default='two',help="features to select")
    
    args = parser.parse_args()
    print >>sys.stderr,args 
    if args.rs:
        random_state = int(args.rs)
    if random_state:
        random.seed(random_state)
        np.random.seed(random_state)

    if args.cmd == 'test':
        test(args)
    elif args.cmd == 'make':
        game_analysis(args.fn,args.clf,sel=args.sel,toest=True)
    else:
        raise ValueError("bad cmd")
    
