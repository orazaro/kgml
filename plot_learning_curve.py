#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   scikit-learn developers
# Author:   Oleg Razgulyaev
# License:  BSD 3 clause
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#example-model-selection-plot-learning-curve-py
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#example-model-selection-plot-validation-curve-py
"""
===============
Learning Curves
===============
A learning curve shows the validation and training score of an estimator for
varying numbers of training samples. It is a tool to find out how much we
benefit from adding more training data and whether the estimator suffers more
from a variance error or a bias error. If both the validation score and the
training score converge to a value that is too low with increasing size of the
training set, we will not benefit much from more training data. If the
training score is much greater than the validation score for the maximum
number of training samples, adding more training samples will most likely
increase generalization.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5),
                        scoring=None, ax=None):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    if ax is None:
        fig, ax1 = plt.subplots()
    else:
        ax1 = ax
    ax1.set_title(title)
    if ylim is not None:
        ax1.yset_lim(*ylim)
    sunits = '%' if isinstance(train_sizes[0], np.float64) else 'samples'
    ax1.set_xlabel("Training examples ({})".format(sunits))
    ax1.set_ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax1.grid()

    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    ax1.legend(loc="best")
    return ax1


def plot_validation_curve(
        estimator, X, y, param_name, param_range, title="Validation Curve",
        ylim=None, semilog=False,
        cv=None, n_jobs=1, scoring=None, ax=None):
    # param_range = np.logspace(-6, -1, 5)
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if ax is None:
        fig, ax1 = plt.subplots()
    else:
        ax1 = ax
    ax1.set_title(title)
    ax1.set_xlabel(param_name)
    ax1.set_ylabel("Score")
    ax1.grid()
    if ylim is not None:
        ax1.set_ylim(ylim)
    if semilog:
        ax1.semilogx(param_range, train_scores_mean, 'o-',
                     label="Training score",
                     color="r")
    else:
        ax1.plot(param_range, train_scores_mean, 'o-', label="Training score",
                 color="r")
    ax1.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="r")
    if semilog:
        ax1.semilogx(param_range, test_scores_mean, 'o-',
                     label="Cross-validation score",
                     color="g")
    else:
        ax1.plot(param_range, test_scores_mean, 'o-',
                 label="Cross-validation score",
                 color="g")
    ax1.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    ax1.legend(loc="best")
    return ax1

"""
==========================
Plotting Validation Curves
==========================
In this plot you can see the training scores and validation scores of an SVM
for different values of the kernel parameter gamma. For very low values of
gamma, you can see that both the training score and the validation score are
low. This is called underfitting. Medium values of gamma will result in high
values for both scores, i.e. the classifier is performing fairly well. If gamma
is too high, the classifier will overfit, which means that the training score
is good but the validation score is poor.
"""


def test_plot_validation_curve():
    from sklearn.datasets import load_digits
    from sklearn.svm import SVC
    digits = load_digits()
    X, y = digits.data, digits.target
    estimator = SVC()
    param_range = np.logspace(-6, -1, 5)
    plot_validation_curve(
            estimator, X, y, param_name="gamma",
            param_range=param_range,
            title="Validation Curve for SVC", ylim=None, cv=None, n_jobs=-1,
            scoring=None, ax=None)
    plt.show()


"""
========================
Plotting Learning Curves
========================
On the left side the learning curve of a naive Bayes classifier is shown for
the digits dataset. Note that the training score and the cross-validation score
are both not very good at the end. However, the shape of the curve can be found
in more complex datasets very often: the training score is very high at the
beginning and decreases and the cross-validation score is very low at the
beginning and increases. On the right side we see the learning curve of an SVM
with RBF kernel. We can see clearly that the training score is still around
the maximum and the validation score could be increased with more training
samples.
"""


def test_plot_learning_curve():
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.datasets import load_digits

    digits = load_digits()
    X, y = digits.data, digits.target

    title = "Learning Curves (Naive Bayes)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation
    # set.
    cv = cross_validation.ShuffleSplit(digits.data.shape[0], n_iter=100,
                                       test_size=0.2, random_state=0)

    estimator = GaussianNB()
    plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv,
                        n_jobs=4)

    title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = cross_validation.ShuffleSplit(digits.data.shape[0], n_iter=10,
                                       test_size=0.2, random_state=0)
    estimator = SVC(gamma=0.001)
    plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

    plt.show()

if __name__ == '__main__':
    test_plot_validation_curve()
