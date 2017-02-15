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
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_array, check_consistent_length


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


def test(args):
    # test_get_confusion_matrix()
    test_log_loss_array()
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
