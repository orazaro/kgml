"""
==============================
Probability Calibration curves
==============================

When performing classification one often wants to predict not only the class
label, but also the associated probability. This probability gives some
kind of confidence on the prediction. This example demonstrates how to display
how well calibrated the predicted probabilities are and how to calibrate an
uncalibrated classifier.

The experiment is performed on an artificial dataset for binary classification
with 100.000 samples (1.000 of them are used for model fitting) with 20
features. Of the 20 features, only 2 are informative and 10 are redundant. The
first figure shows the estimated probabilities obtained with logistic
regression, Gaussian naive Bayes, and Gaussian naive Bayes with both isotonic
calibration and sigmoid calibration. The calibration performance is evaluated
with Brier score, reported in the legend (the smaller the better). One can
observe here that logistic regression is well calibrated while raw Gaussian
naive Bayes performs very badly. This is because of the redundant features
which violate the assumption of feature-independence and result in an overly
confident classifier, which is indicated by the typical transposed-sigmoid
curve.

Calibration of the probabilities of Gaussian naive Bayes with isotonic
regression can fix this issue as can be seen from the nearly diagonal
calibration curve. Sigmoid calibration also improves the brier score slightly,
albeit not as strongly as the non-parametric isotonic regression. This can be
attributed to the fact that we have plenty of calibration data such that the
greater flexibility of the non-parametric model can be exploited.

The second figure shows the calibration curve of a linear support-vector
classifier (LinearSVC). LinearSVC shows the opposite behavior as Gaussian
naive Bayes: the calibration curve has a sigmoid curve, which is typical for
an under-confident classifier. In the case of LinearSVC, this is caused by the
margin property of the hinge loss, which lets the model focus on hard samples
that are close to the decision boundary (the support vectors).

Both kinds of calibration can fix this issue and yield nearly identical
results. This shows that sigmoid calibration can deal with situations where
the calibration curve of the base classifier is sigmoid (e.g., for LinearSVC)
but not where it is transposed-sigmoid (e.g., Gaussian naive Bayes).
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#         Oleg Razgulyaev <oleg@razgulyaev.com>
# License: BSD Style.
# http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#example-calibration-plot-calibration-curve-py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import clone
from sklearn.externals.joblib import Parallel, delayed

def calibration_curve_nan(y_true, y_prob, n_bins=5, n_power=1, minsamples=0, bins=None):
    """ Compute true and predicted probabilities for a calibration curve.
    
    For the empty bins insert np.nan but do not skip the bin as is done in the 
    sklearn version of the same function.

    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.
    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.
    n_bins : int
        Number of bins. A bigger number requires more data.
    n_power: int, optional (default=1)
        make increasing sizes of bins to be useful for an imbalanced datasets 
    minsamples: int, optional (default=0)
        min number of samples in bid
        if minsamples > 0:
            will collapse adjacent bids with low number of samples starting from right
    bins: array, shape (n_bids+1,), optional (default=None)
        bins margins to use
        if=None: build new
        else: use this margins

    Returns
    -------
    prob_true : array, shape (n_bins+1,)
        The true probability in each bin (fraction of positives).
    prob_pred : array, shape (n_bins+1,)
        The mean predicted probability in each bin.
    bins: array
        bins margins used
    
    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).
    """
    if bins is None:
        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
        if minsamples > 0:
            removed = []
            assert np.min(y_prob) >= bins[0] and np.max(y_prob) <= bins[-1]
            #print bins
            n = len(bins)
            i,j = n-2,n-1
            while i>0 and j>0:
                k = np.sum((bins[i]<=y_prob) & (y_prob<bins[j]))
                #print i,j,bins[i],bins[j],k
                if k < minsamples:
                    removed.append(i)
                    i -= 1
                else:
                    j = i
                    i = j - 1
            #print removed
            bins = np.delete(bins, removed)
            #print bins
        else:
            bins = np.power(bins,n_power)

         
    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    prob_true = [e1/e2 if e2 != 0 else np.nan for (e1,e2) in zip(bin_true,bin_total)]
    prob_pred = [e1/e2 if e2 != 0 else np.nan for (e1,e2) in zip(bin_sums,bin_total)]

    n_bins = len(bins)-1
    return np.array(prob_true), np.array(prob_pred), bins, n_bins

def calibration_inner_loop(clf,X,y,train,test,n_bins,n_power,bins_used,minsamples):
    X_train, y_train  = X[train],y[train]
    X_test, y_test = X[test],y[test]
    
    clf.fit(X_train, y_train)
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):  # use decision function
        prob_pos = clf.decision_function(X_test)
        y_proba = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    else:
        raise RuntimeError("clf without predict_proba or decision_function")

    fraction_of_positives, mean_predicted_value, bins_used, n_bins = \
        calibration_curve_nan(y_test, y_proba, n_bins=n_bins, n_power=n_power, 
            bins=bins_used, minsamples=minsamples)
    #print fraction_of_positives.shape, mean_predicted_value.shape
    return (\
        np.array(list(fraction_of_positives)+list(mean_predicted_value)),
        brier_score_loss(y_test, y_proba, pos_label=y_test.max()),
        metrics.roc_auc_score(y_test, y_proba),
        bins_used, n_bins
        )

def plot_calibration_curve_boot(X, y, clfs, names=None, n_bins=10, n_power=1, n_iter=10, n_jobs=1, fig_index=1, scatt=False, minsamples=50):
    """ Plot calibration curve for est w/o and with calibration. 
    
        Use bootstrap for the cross-validation

        Parameters
        ----------
        X: array, shape(n_samples, n_features)
            dataset
        y: array, shape(n_samples,)
            targets
        clfs: list of estimators or one estimator
            estimator for which plot the calibration curve
        names: list, optional (default=None)
            names of estimators
            if==None: try to use est.name
        n_bins: int, optional (default=10)
            number of bins for the calibration
        n_power: int, optional (default=1)
            power to squeeze bin margins towards zero
        n_iter: int, optional (default=10)
            number of cv iterations
        n_jobs: int, optional (default=1)
            number of processor cores to use
        fig_index: int, optional (default=1)
            fig number if any
        scatt: bool, optional (default=False)
            plot all measurements as scatterplot
            if scatt==False: add confidence intervals as errorbars
        minsamples: int, optional (default=50)
            min number of samples in bin
            if minsamples > 0:
                will combine adjacent bids with low number of samples in one 
                starting from right
    """
    import sklearn.cross_validation as cross_validation
    from sklearn import (metrics, cross_validation)
    from modsel import bootstrap_632
   
    if not isinstance(clfs, (list,tuple,set)):
        clfs = [clfs]

    fig,ax1 = plt.subplots(1,1,figsize=(10, 7))

    ax1.plot([0,0],[-0.05,1.05],'k--',lw=1, label="Borders of bins")
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    bins_used = None
    for i_clf,clf in enumerate(clfs):
        if names is not None:
            name = names[i_clf]
        elif hasattr(clf,'name'):
            name = clf.name
        else:
            name = "Undef"
        Res,clf_score,clf_auc = [],0.0,0.0
        cv = list(bootstrap_632(len(y), n_iter))
        if bins_used is None:
            train,test = cv[0]
            (_,_,_,bins_used, n_bins) = calibration_inner_loop(clf,X,y,train,test,n_bins,
                n_power,bins_used,minsamples)
        if n_jobs != 1:
            verbose=0
            pre_dispatch='2*n_jobs'
            parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
            r = parallel(
                delayed(calibration_inner_loop)(clone(clf),X,y,train,test,n_bins,n_power,
                                    bins_used,minsamples)
                for train, test in cv)
        else:
            r = []
            for train,test in cv:
                r.append( calibration_inner_loop(clf,X,y,train,test,n_bins,n_power,
                    bins_used,minsamples) )
        for (r1,r2,r3,_,_) in r:
            Res.append(r1)
            clf_score += r2
            clf_auc += r3
            
        clf_score /= n_iter
        clf_auc /= n_iter
        Res = np.array(Res)
        #print "Res:",Res.shape
        Res_mean = np.nanmean(Res,axis=0)
        Res_err = np.nanstd(Res,axis=0)*1.96
        #print "Res_mean:",Res_mean
        #print "Res_err:",Res_err 
        
        y1 = Res_mean[:(n_bins+1)][:n_bins]
        y1err = Res_err[:(n_bins+1)][:n_bins]
        x1 = Res_mean[(n_bins+1):][:n_bins]
        x1err = Res_err[(n_bins+1):][:n_bins]

        if len(clfs) > 1:
            ax1.plot(x1, y1, "s-",
                 label="%s (brier=%1.3f, auc=%1.3f)" % (name, clf_score, clf_auc))
        else:
            if scatt:
                for irow in range(Res.shape[0]):
                    x2 = Res[irow,(n_bins+1):][:n_bins]
                    y2 = Res[irow,:(n_bins+1)][:n_bins]
                    ax1.scatter(x2,y2,alpha=0.5)
                    x1err = y1err = None
            ax1.errorbar(x1, y1, marker='o', xerr=x1err, yerr=y1err, ls='--', lw=2,
                label="%s (brier=%1.3f, auc=%1.3f)" % (name, clf_score, clf_auc) )

    # draw bins margins
    for x_bin in bins_used[1:-1]:
        ax1.plot([x_bin,x_bin],[-0.05,1.05],'k--',lw=1)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="upper left")
    ax1.set_title('Calibration plots  (reliability curve)')
    ax1.set_xlabel("Mean predicted value")

    plt.tight_layout()

def plot_calibration_curve_cv(X, y, est, name, bins=10, n_folds=8, n_jobs=1, fig_index=1):
    """Plot calibration curve for est w/o and with calibration. """
    import sklearn.cross_validation as cross_validation
    from sklearn import (metrics, cross_validation)
    from model_selection import cross_val_predict_proba
    
    # Calibrated with isotonic calibration
    cv = 2
    isotonic = CalibratedClassifierCV(est, cv=cv, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=cv, method='sigmoid')

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        
        y_true = y
        scoring = 'roc_auc'
        cv1 = cross_validation.StratifiedKFold(y,n_folds)
        y_proba, scores = cross_val_predict_proba(clf, X, y, scoring=scoring, 
            cv=cv1, n_jobs=n_jobs, verbose=0, fit_params=None, pre_dispatch='2*n_jobs')
        y_pred = np.array(y_proba>0.5,dtype=int)

        clf_score = brier_score_loss(y_true, y_proba, pos_label=y_true.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_true, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_true, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_true, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_true, y_proba, n_bins=bins)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(y_proba, range=(0, 1), bins=bins, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()

def plot_calibration_curve_old(X_train, X_test, y_train, y_test, y, est, name, fig_index, bins=10):
    """Plot calibration curve for est w/o and with calibration. """
    from sklearn.linear_model import LogisticRegression
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1., solver='lbfgs')

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=bins)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=bins, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()

plot_calibration_curve = plot_calibration_curve_boot

#--- tests ---------#

def test_calibration_curve_nan():
    
    y_true = np.arange(20)
    y_prob = np.square(np.arange(20)/20.)
    print y_prob
    calibration_curve_nan(y_true, y_prob, n_bins=5, minsamples=3)
    calibration_curve_nan(y_true, y_prob, n_bins=5, minsamples=5)
    calibration_curve_nan(y_true, y_prob, n_bins=5, minsamples=9)

def test_plot_calibration_curve():
    from sklearn import datasets
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import LinearSVC
    from sklearn.cross_validation import train_test_split
    X, y = datasets.make_classification(n_samples=100000, n_features=20,
                                        n_informative=2, n_redundant=10,
                                        random_state=42)
    # Plot calibration cuve for Gaussian Naive Bayes
    plot_calibration_curve(X, y, GaussianNB(), "Naive Bayes", fig_index=1 )
    plt.show()

def test_plot_calibration_curve_old():
    from sklearn import datasets
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import LinearSVC
    from sklearn.cross_validation import train_test_split
    print(__doc__)
    # Create dataset of classification task with many redundant and few
    # informative features
    X, y = datasets.make_classification(n_samples=100000, n_features=20,
                                        n_informative=2, n_redundant=10,
                                        random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99,
                                                        random_state=42)
    # Plot calibration cuve for Gaussian Naive Bayes
    plot_calibration_curve_old(X_train, X_test, y_train, y_test, y, GaussianNB(), "Naive Bayes", 1)
    # Plot calibration cuve for Linear SVC
    plot_calibration_curve_old(X_train, X_test, y_train, y_test, y, LinearSVC(), "SVC", 2)
    plt.show()

if __name__ == '__main__':
    #test_plot_calibration_curve_old()
    #test_plot_calibration_curve()
    test_calibration_curve_nan()
