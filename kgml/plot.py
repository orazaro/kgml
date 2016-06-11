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
    
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    
    nb = 20
    bx = (x_max-x_min)/nb
    by = (y_max-y_min)/nb
    
    x_min, x_max = x_min - bx, x_max + bx
    y_min, y_max = y_min - by, y_max + by
    
    nh = 300
    hx = (x_max-x_min)/nh
    hy = (y_max-y_min)/nh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx),
                         np.arange(y_min, y_max, hy))
    
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


def plot_roc_crossval(model, X, y, n_folds=6, figsize=(6,6),show_folds=False):
    """ Run classifier with cross-validation and plot ROC curves
        from http://goo.gl/NMhWvf
    """
    from sklearn.cross_validation import StratifiedKFold
    from scipy import interp
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt    

    cv = StratifiedKFold(y, n_folds=n_folds)

    plt.figure(figsize=figsize)
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
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_roc_curve(y_true, y_proba, ax=None, figsize=(6,6)):
    """ Run classifier with cross-validation and plot ROC curves
        from http://goo.gl/NMhWvf
    """
    from sklearn.metrics import roc_curve, auc

    if ax is None:
        fig,ax1 = plt.subplots(1,1,figsize=figsize)
    else: 
        ax1 = ax

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    ax1.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    tpr[-1] = 1.0
    auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, 'k--',
             label='ROC (area = %0.2f)' % auc, lw=2)

    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver operating characteristic')
    ax1.legend(loc="lower right")
    if ax is None:
        plt.show()

def plot_pred_proba_hist_class(y_true, y_proba, ax, cl, bins=20):
    if cl > 0:
        yc = y_proba[y_true>0]
    else:
        yc = y_proba[y_true<=0]
    ax.hist(yc,bins=bins); 
    ax.set_title("class {}".format(cl))
    ax.set_xlim((0,1))

def plot_pred_proba_hist(y_true, y_proba, bins=20, figsize=(12,5)):
    fig,axarr = plt.subplots(1,2,figsize=figsize)
    axgen = (e for e in np.array(axarr).ravel())
    
    plot_pred_proba_hist_class(y_true, y_proba, ax=axgen.next(), cl=1, bins=bins)
    plot_pred_proba_hist_class(y_true, y_proba, ax=axgen.next(), cl=2, bins=bins)

    plt.suptitle("Predicted probability distributions",fontsize=16)
    #plt.figtext(.02, -.10, "This is text on the bottom of the figure.\nHere I've made extra room for adding more text.\n" + ("blah "*16+"\n")*3)

def plot_pred_proba_distrib(y_true, y_proba, ax=None, figsize=(8,5)):
    if ax is None:
        fig,ax1 = plt.subplots(1,1,figsize=figsize)
    else:
        ax1 = ax

    y0 = y_proba[y_true<=0]
    y1 = y_proba[y_true>0]
    
    from scipy.stats import gaussian_kde 
    density0 = gaussian_kde( y0 )
    density1 = gaussian_kde( y1 )

    x = np.arange(0., 1, .01)
    ax1.plot(x, density0(x),label="class 0",lw=3)
    ax1.plot(x, density1(x),label="class 1",lw=3)
    ax1.legend()
    ax1.set_title("probability density functions")
   
def plot_estimates(y_true, y_proba, bins=20, figsize=(8,5), tohist=True):
    nx = 2 if tohist else 1
    fig,axarr = plt.subplots(nx,2,figsize=figsize)
    axgen = (e for e in np.array(axarr).ravel())

    if tohist:
        plot_pred_proba_hist_class(y_true, y_proba, ax=axgen.next(), cl=1, bins=bins)
        plot_pred_proba_hist_class(y_true, y_proba, ax=axgen.next(), cl=0, bins=bins)
    plot_pred_proba_distrib(y_true, y_proba, ax=axgen.next())
    plot_roc_curve(y_true, y_proba, ax=axgen.next())
    
    plt.suptitle("Predicted probability distributions",fontsize=14)


def plot_confusion_matrix(y_test, y_pred, target_names,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """ Calculate and plot confusion_matrix. """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return cm
