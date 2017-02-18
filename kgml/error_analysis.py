#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev
# License:  BSD 3 clause
"""
    Cluster  analysis
"""
from __future__ import (division, print_function)
import sys
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, hinge_loss
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_array, check_consistent_length, column_or_1d


RANDOM_STATE = 1


# --------- Tests -------------------------------------#


def get_confusion_matrix(y_true, y_pred, class_names=None,
                         normalize=False,
                         to_plot=True,
                         title='Confusion matrix',
                         cmap=plt.cm.Blues,
                         fmt=None,
                         figsize=None,
                         thresh=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # np.set_printoptions(
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')
    # print(cm)
    if to_plot:
        if figsize:
            plt.figure(figsize=figsize)
        if fmt is None:
            fmt = '{:.1f}' if normalize else '{:.0f}'
        if class_names is None:
            class_names = ['class{}'.format(i + 1)
                           for i in range(len(np.unique(y_true)))]
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        if thresh is None:
            thresh = cm.max() / 1.29
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, fmt.format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    return cm


def test_get_confusion_matrix():
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, n_redundant=2,
        n_repeated=0, n_classes=9, n_clusters_per_class=2)
    clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    y_pred = cross_val_predict(clf, X, y, cv=5, method='predict')
    cnf_matrix = get_confusion_matrix(
        y, y_pred, class_names=None, normalize=True,
        title='Confusion matrix, with normalization')
    plt.show()
    print(cnf_matrix)

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Arnaud Joly <a.joly@ulg.ac.be>
#          Jochen Wersdorfer <jochen@wersdoerfer.de>
#          Lars Buitinck
#          Joel Nothman <joel.nothman@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Jatin Shah <jatindshah@gmail.com>
#          Saurabh Jha <saurabh.jhaa@gmail.com>
#          Bernardo Stein <bernardovstein@gmail.com>
# License: BSD 3 clause


def log_loss_array(y_true, y_pred, eps=1e-15, labels=None):
    """ LogLoss for individual samples
        Copied from https://goo.gl/uy3oFZ
    """
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_pred, y_true)

    lb = LabelBinarizer()

    if labels is not None:
        lb.fit(labels)
    else:
        lb.fit(y_true)

    if len(lb.classes_) == 1:
        if labels is None:
            raise ValueError('y_true contains only one label ({0}). Please '
                             'provide the true labels explicitly through the '
                             'labels argument.'.format(lb.classes_[0]))
        else:
            raise ValueError('The labels array needs to contain at least two '
                             'labels for log_loss, '
                             'got {0}.'.format(lb.classes_))

    transformed_labels = lb.transform(y_true)

    if transformed_labels.shape[1] == 1:
        transformed_labels = np.append(1 - transformed_labels,
                                       transformed_labels, axis=1)

    # Clipping
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # If y_pred is of single dimension, assume y_true to be binary
    # and then check.
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)

    # Check if dimensions are consistent.
    transformed_labels = check_array(transformed_labels)
    if len(lb.classes_) != y_pred.shape[1]:
        if labels is None:
            raise ValueError("y_true and y_pred contain different number of "
                             "classes {0}, {1}. Please provide the true "
                             "labels explicitly through the labels argument. "
                             "Classes found in "
                             "y_true: {2}".format(transformed_labels.shape[1],
                                                  y_pred.shape[1],
                                                  lb.classes_))
        else:
            raise ValueError('The number of classes in labels is different '
                             'from that in y_pred. Classes found in '
                             'labels: {0}'.format(lb.classes_))

    # Renormalize
    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
    loss = -(transformed_labels * np.log(y_pred)).sum(axis=1)

    return loss


def test_log_loss_array():
    X, y = make_classification(
        n_samples=10000, n_features=20, n_informative=10, n_redundant=2,
        n_repeated=0, n_classes=9, n_clusters_per_class=2)
    clf = LogisticRegression()
    y_pred_proba = cross_val_predict(clf, X, y, cv=4, n_jobs=1,
                                     method='predict_proba')

    lla = log_loss_array(y, y_pred_proba, eps=1e-15, labels=None)
    print("lla:", lla.mean(), lla)


def hinge_loss_array(y_true, pred_decision, labels=None):
    """hinge loss (non-regularized)
    Copied from https://goo.gl/uy3oFZ
    In binary class case, assuming labels in y_true are encoded with +1 and -1,
    when a prediction mistake is made, ``margin = y_true * pred_decision`` is
    always negative (since the signs disagree), implying ``1 - margin`` is
    always greater than 1.  The cumulated hinge loss is therefore an upper
    bound of the number of mistakes made by the classifier.
    In multiclass case, the function expects that either all the labels are
    included in y_true or an optional labels argument is provided which
    contains all the labels. The multilabel margin is calculated according
    to Crammer-Singer's method. As in the binary case, the cumulated hinge loss
    is an upper bound of the number of mistakes made by the classifier.
    Read more in the :ref:`User Guide <hinge_loss>`.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True target, consisting of integers of two values. The positive label
        must be greater than the negative label.
    pred_decision : array, shape = [n_samples] or [n_samples, n_classes]
        Predicted decisions, as output by decision_function (floats).
    labels : array, optional, default None
        Contains all the labels for the problem. Used in multiclass hinge loss.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    -------
    loss : float
    References
    ----------
    .. [1] `Wikipedia entry on the Hinge loss
           <https://en.wikipedia.org/wiki/Hinge_loss>`_
    .. [2] Koby Crammer, Yoram Singer. On the Algorithmic
           Implementation of Multiclass Kernel-based Vector
           Machines. Journal of Machine Learning Research 2,
           (2001), 265-292
    .. [3] `L1 AND L2 Regularization for Multiclass Hinge Loss Models
           by Robert C. Moore, John DeNero.
           <http://www.ttic.edu/sigml/symposium2011/papers/
           Moore+DeNero_Regularization.pdf>`_
    """
    check_consistent_length(y_true, pred_decision)
    pred_decision = check_array(pred_decision, ensure_2d=False)
    y_true = column_or_1d(y_true)
    y_true_unique = np.unique(y_true)
    if y_true_unique.size > 2:
        if (labels is None and pred_decision.ndim > 1 and
                (np.size(y_true_unique) != pred_decision.shape[1])):
            raise ValueError("Please include all labels in y_true "
                             "or pass labels as third argument")
        if labels is None:
            labels = y_true_unique
        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        mask = np.ones_like(pred_decision, dtype=bool)
        mask[np.arange(y_true.shape[0]), y_true] = False
        margin = pred_decision[~mask]
        margin -= np.max(pred_decision[mask].reshape(y_true.shape[0], -1),
                         axis=1)

    else:
        # Handles binary class case
        # this code assumes that positive and negative labels
        # are encoded as +1 and -1 respectively
        pred_decision = column_or_1d(pred_decision)
        pred_decision = np.ravel(pred_decision)

        lbin = LabelBinarizer(neg_label=-1)
        y_true = lbin.fit_transform(y_true)[:, 0]

        try:
            margin = y_true * pred_decision
        except TypeError:
            raise TypeError("pred_decision should be an array of floats.")

    losses = 1 - margin
    # The hinge_loss doesn't penalize good enough predictions.
    losses[losses <= 0] = 0
    return losses


def test_hinge_loss_array(n_classes=9):
    X, y = make_classification(
        n_samples=10000, n_features=20, n_informative=10, n_redundant=2,
        n_repeated=0, n_classes=n_classes, n_clusters_per_class=2)
    clf = LogisticRegression()
    y_pred_proba = cross_val_predict(clf, X, y, cv=4, n_jobs=1,
                                     method='predict_proba')

    if n_classes == 2:
        y_pred_proba = y_pred_proba[:, 1]
    lla = hinge_loss_array(y, y_pred_proba, labels=None)
    print("hinge_loss:", hinge_loss(y, y_pred_proba))
    print("lla:", lla.mean(), lla)


def test(args):
    # test_get_confusion_matrix()
    # test_log_loss_array()
    # test_hinge_loss_array()
    test_hinge_loss_array(n_classes=2)
    print("Test OK", file=sys.stderr)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Commands.')
    parser.add_argument('cmd', nargs='?',
                        default='test', help="make_train|make_test")
    parser.add_argument('-rs', type=int, default=None, help="random_state")

    args = parser.parse_args()
    print(args, file=sys.stderr)
    if args.rs:
        RANDOM_STATE = int(args.rs)
    if RANDOM_STATE:
        random.seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

    if args.cmd == 'test':
        test(args)
    else:
        raise ValueError("bad cmd")
