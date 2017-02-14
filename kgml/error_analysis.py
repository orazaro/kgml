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
import numpy as np
import matplotlib.pyplot as plt

RANDOM_STATE = 1


# --------- Tests -------------------------------------#


def get_confusion_matrix(y_true, y_pred, class_names=None,
                         normalize=False,
                         to_plot=True,
                         title='Confusion matrix',
                         cmap=plt.cm.Blues,
                         fmt=None,
                         figsize=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from sklearn.metrics import confusion_matrix
    import itertools
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

        thresh = cm.max() / 1.27
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, fmt.format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    return cm


def test_get_confusion_matrix():
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict

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


def test(args):
    test_get_confusion_matrix()
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
