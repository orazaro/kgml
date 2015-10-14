#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
    Plot data
"""
from __future__ import division
import numpy as np

import matplotlib.pyplot as plt
#from matplotlib import style
#style.use("ggplot")

def rand_jitter(arr, dx=.003):
    stdev = dx*(max(arr)-min(arr))
    if stdev == 0.0:
        stdev = dx
    return arr + np.random.randn(len(arr)) * stdev

def plt_subplots(nrow=1, ncol=1, sharex=False, sharey=False, figsize=(10,10)):
    """ Initialize subplots. """
    fig,axarr = plt.subplots(nrow, ncol, sharex=sharex, sharey=sharey,figsize=figsize)
    axgen = (e for e in np.array(axarr).ravel())
    return fig,axgen

def plot_decision_line(clf, X, y, names=None):
    """ Plot decision line

    Plot decision line for the linear model (svm.SVC(kernel= "linear"))
    
    Parameters
    ----------
    clf: (BaseEstimator, ClassifierMixin)
        classifier

    X: array-like, shape=(n_samples,n_features) 
        train data

    y: array-like, shape=(n_samples,)
        labels

    Returns:
    --------
    None
    """
    w= clf.coef_[0]
    w0 = w[0]
    w1 = w[1]
    intercept = clf.intercept_[0]
    #print "w:",(w0,w1),"intercept:",intercept

    a = -w0 / w1
    xx = np.linspace(min(X[:,0]), max(X[:,0]))
    yy = a * xx - intercept / w1

    r = plt.plot(xx,yy, "k-", label="non weighted")
    plt.scatter(X[:,0], X[:,1], c = y)
    plt.ylabel(names[1])
    plt.xlabel(names[0])
    plt.show()

def plot_decision_boundary(clf, X, y, ax, sample_weight=None, names=None, title="Decision Boundary"):
    """ Plot decision boundary.

    Plot decision boundary for any model with the method predict working

    Parameters
    ----------
    clf: (BaseEstimator, ClassifierMixin)
        classifier

    sample_weight: array-like, shape=(n_samples,), optional (default=None)
        weight of the samples

    X: array-like, shape=(n_samples,n_features) 
        train data
        must have 2 columns only

    y: array-like, shape=(n_samples,)
        labels

    title: str
        title of the graph

    Returns:
    --------
    None
    """
    assert X.shape[1] == 2
    if sample_weight is None:
        sample_weight = np.ones(X.shape[0])
    h = .02  # step size in the mesh
    b = h*50
    
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - b, X[:, 0].max() + b
    y_min, y_max = X[:, 1].min() - b, X[:, 1].max() + b
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z>0.5

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30*sample_weight, alpha=0.9, cmap=plt.cm.Paired)
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    #ax.xticks(())
    #ax.yticks(())
    ax.set_title(title)
    #ax.legend()


def plot_roc_crossval(model, X, y, show_folds=False):
    """ Run classifier with cross-validation and plot ROC curves
        from http://goo.gl/NMhWvf
    """
    from sklearn.cross_validation import StratifiedKFold
    from scipy import interp
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt    

    cv = StratifiedKFold(y, n_folds=6)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas_ = model.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        if show_folds:
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


