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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

def calibration_curve_my(y_true, y_prob, n_bins=5):
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    bins = np.power(bins,3)
    bins = np.array([0.,0.1,0.2,0.35,0.5,1.+1e-8])
    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    prob_true = [e1/e2 if e2 != 0 else np.nan for (e1,e2) in zip(bin_true,bin_total)]
    prob_pred = [e1/e2 if e2 != 0 else np.nan for (e1,e2) in zip(bin_sums,bin_total)]

    return np.array(prob_true), np.array(prob_pred)

def plot_calibration_curve_boot(X, y, est, name, bins=10, n_iter=100, n_jobs=1, fig_index=1):
    """ Plot calibration curve for est w/o and with calibration. 
        using bootstrap
    """
    import sklearn.cross_validation as cross_validation
    from sklearn import (metrics, cross_validation)
    from modsel import bootstrap_632
    
    # Calibrated with isotonic calibration
    cv = 10
    cv = bootstrap_632(len(y), 50)
    isotonic = CalibratedClassifierCV(est, cv=cv, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=cv, method='sigmoid')

    est1 = CalibratedClassifierCV(est, cv=cv, method='isotonic')


    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    clfs = [(est, name),
            (isotonic, name + ' + Isotonic'),
            (sigmoid, name + ' + Sigmoid')][:2]
    for clf, name in clfs:
        Res,clf_score = [],0
        cv = bootstrap_632(len(y), n_iter)
        for train,test in cv:
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
        
            fraction_of_positives, mean_predicted_value = \
                calibration_curve_my(y_test, y_proba, n_bins=bins)
            #print fraction_of_positives.shape, mean_predicted_value.shape
            Res.append(np.array(list(fraction_of_positives)+list(mean_predicted_value)))
            clf_score += brier_score_loss(y_test, y_proba, pos_label=y_test.max())
            
        clf_score /= n_iter
        Res = np.array(Res)
        print "Res:",Res.shape
        Res_mean = np.nanmean(Res,axis=0)
        Res_err = np.nanstd(Res,axis=0)*1.96
        #print "Res_mean:",Res_mean
        #print "Res_err:",Res_err 
        
        y1 = Res_mean[:(bins+1)][:bins]
        y1err = Res_err[:(bins+1)][:bins]
        x1 = Res_mean[(bins+1):][:bins]
        x1err = Res_err[(bins+1):][:bins]

        if False:
            y_pred = np.array(y_proba>0.5,dtype=int)

            clf_score = brier_score_loss(y_true, y_proba, pos_label=y_true.max())
            print("%s:" % name)
            print("\tBrier: %1.3f" % (clf_score))
            print("\tPrecision: %1.3f" % precision_score(y_true, y_pred))
            print("\tRecall: %1.3f" % recall_score(y_true, y_pred))
            print("\tF1: %1.3f\n" % f1_score(y_true, y_pred))


        if len(clfs) > 1:
            ax1.plot(x1, y1, "s-",
                 label="%s (%1.3f)" % (name, clf_score))
        else:
            ax1.errorbar(x1, y1, marker='o', xerr=x1err, yerr=y1err, ls='--', lw=2,
                label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(y_proba, range=(0, 1), bins=bins, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="upper left")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

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
    test_plot_calibration_curve()
