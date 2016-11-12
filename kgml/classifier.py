#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev
# License:  BSD 3 clause
"""
    Classifiers
"""
from __future__ import division, print_function

import sys
import random
import numpy as np
import logging

from sklearn.base import BaseEstimator, ClassifierMixin

import sklearn.linear_model as lm
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,)
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

from sklearn import model_selection
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets.samples_generator import (make_classification, )
from sklearn import (metrics, model_selection)

# kgml
from imbalanced import round_down
from modsel import bootstrap_632
from paramsel import param_grids
from modsel import (precision_sensitivity_specificity, best_threshold)

logger = logging.getLogger(__name__)


def get_clf(cl, n_jobs=1, random_state=0, class_weight='balanced'):
    """ Select clasifier by name
    """
    lm1 = {'C': [0.0001, 0.001, 0.01, 0.1, 0.3, 1, 3, 10]}
    C_range = 10.0 ** np.arange(-5, 3)
    C_range = np.hstack([C_range, [0.3, 3]])
    # lm2 = dict(C=C_range)
    rf1 = {'max_depth': [2, 4, 8, 16, 24, 32]}

    if cl == 'rf2':
        clf = RandomForestClassifier(
                n_estimators=100, min_samples_leaf=1,
                max_features='auto', class_weight=class_weight,
                n_jobs=n_jobs, random_state=random_state, verbose=0)
    elif cl == 'rf':
        clf1 = RandomForestClassifier(
                n_estimators=100, max_depth=2,
                max_features='auto', class_weight=class_weight,
                n_jobs=n_jobs, random_state=random_state, verbose=0)
        clf = model_selection.GridSearchCV(clf1, rf1, cv=4, n_jobs=n_jobs,
                                       verbose=0)
    elif cl == 'dt':
        from sklearn.tree import DecisionTreeClassifier
        clf1 = DecisionTreeClassifier(max_depth=2, max_features='auto',
                                      class_weight=class_weight)
        clf = model_selection.GridSearchCV(clf1, rf1, cv=4, n_jobs=n_jobs,
                                       verbose=0)
    elif cl == 'lr2':
        clf = lm.LogisticRegression(
                    penalty='l2', dual=True, tol=0.0001,
                    C=1, fit_intercept=True, intercept_scaling=1.0,
                    class_weight=class_weight, random_state=random_state)

    elif cl == 'lr1':
        clf = lm.LogisticRegression(
                    penalty='l1', dual=False, tol=0.0001,
                    C=1.0, fit_intercept=True, intercept_scaling=1.0,
                    class_weight=class_weight, random_state=random_state)
    elif cl == 'lr2g':
        est2 = lm.LogisticRegression(
                    penalty='l2', dual=True, tol=0.0001,
                    C=1, fit_intercept=True, intercept_scaling=1.0,
                    class_weight=class_weight, random_state=random_state)
        clf = model_selection.GridSearchCV(est2, lm1, cv=4, n_jobs=n_jobs,
                                       verbose=0)
    elif cl == 'lr1g':
        est1 = lm.LogisticRegression(
                    penalty='l1', dual=False, tol=0.0001,
                    C=1, fit_intercept=True, intercept_scaling=1.0,
                    class_weight=class_weight, random_state=random_state)
        clf = model_selection.GridSearchCV(est1, lm1, cv=4, n_jobs=n_jobs,
                                       verbose=0)
    elif cl == 'svmL':
        clf = svm.LinearSVC(C=1.0, loss='l2', penalty='l2', dual=True,
                            verbose=0, class_weight=class_weight)
    elif cl == 'svmL1':
        clf = svm.LinearSVC(C=1.0, loss='l2', penalty='l1', dual=False,
                            verbose=0, class_weight=class_weight)
    elif cl == 'svmL2':
        clf = svm.LinearSVC(C=1.0, loss='l1', penalty='l2', verbose=0,
                            class_weight=class_weight)
    elif cl == 'svmL1g':
        # est3 = svm.SVC(kernel='linear',verbose=0)
        est3 = svm.LinearSVC(loss='l2', penalty='l1', dual=False, verbose=0,
                             class_weight=class_weight)
        clf = model_selection.GridSearchCV(est3, lm1, cv=4, n_jobs=n_jobs,
                                       verbose=0)
    elif cl == 'svmL2g':
        # est3 = svm.SVC(kernel='linear',verbose=0)
        est3 = svm.LinearSVC(loss='l1', penalty='l2', verbose=0,
                             class_weight=class_weight)
        clf = model_selection.GridSearchCV(est3, lm1, cv=4, n_jobs=n_jobs,
                                       verbose=0)
    elif cl == 'svmRg':
        # C_range = 10.0 ** np.arange(-2, 9)
        # gamma_range = 10.0 ** np.arange(-5, 4)
        C_range = 10.0 ** np.arange(-3, 4)
        gamma_range = 10.0 ** np.arange(-4, 3)
        svm2 = dict(gamma=gamma_range, C=C_range)
        est3 = svm.SVC(kernel='rbf', verbose=0, class_weight=class_weight)
        clf = model_selection.GridSearchCV(est3, svm2, cv=4, n_jobs=n_jobs,
                                       verbose=0)
    elif cl == 'svmP3':
        # svm1 = {'C':[0.001,0.01,0.1,1.0,10],'gamma':[0.1,0.01,0.001,0.0001]}
        svm3 = {'C': [0.001, 0.01, 0.1, 1.0, 10],
                'gamma': [0.1, 0.01, 0.001, 0.0001],
                'coef0': [0, 1]}
        est4 = svm.SVC(kernel='poly', degree=3, verbose=0,
                       class_weight=class_weight)
        clf = model_selection.GridSearchCV(est4, svm3, cv=4, n_jobs=n_jobs,
                                       verbose=0)
    elif cl == 'mnb':
        clf = MultinomialNB(alpha=1.0)
    elif cl == 'gnb':
        clf = GaussianNB()
    elif cl == 'knn':
        knn1 = {'n_neighbors': 2 ** np.arange(0, 8)}
        clf = model_selection.GridSearchCV(KNeighborsClassifier(), knn1, cv=4,
                                       n_jobs=n_jobs, verbose=0)
    elif cl == 'knn100':
        clf = KNeighborsClassifier(n_neighbors=100)
    elif cl == 'knn1k':
        clf = KNeighborsClassifier(n_neighbors=1024)
    elif cl == 'lda':
        clf = LDA()
    elif cl == 'qda':
        clf = QDA()
    elif cl == 'gb':
        gb1 = {'max_depth': [1, 2, 4, 8],
               'n_estimators': [10, 20, 40, 80, 160]}
        clf = model_selection.GridSearchCV(
            GradientBoostingClassifier(
                learning_rate=0.1,
                random_state=random_state, verbose=0, subsample=1.0),
            gb1, cv=4, n_jobs=n_jobs, verbose=0)
    elif cl == 'rcv':
        clf = RidgeCV_proba()
    elif cl == 'lcv':
        clf = LassoCV_proba()
    elif cl == 'lr':
        clf = LinearRegression_proba()
    else:
        raise ValueError("bad cl:%s" % cl)

    return clf

# ------- Base Classifier Model -----------###


class Model(BaseEstimator):
    """ The base class to inherit from it all our meta estimators.

    Parameters
    ----------
    sclf: str, optional (default=None)
        name of the base estimator if we want to use this class (Model)
        as our meta estimator (do not inherit)
    use_scaler: int, optional (default=2)
        to use standard scaler during transformation phase
    rounddown: bool, optional (default=False)
        round down classes in the training dataset before fitting
    n_jobs:int, optional (default=1)
        number of cores to use to speed up calculations
    rs:int, optional (default=random_state)
        random seed to initialize the random generator
        by default will use the global parameter random_state

    Attributes
    ----------
    Examples
    --------
    References
    ----------
    """
    @property
    def name(self):
        """ The name of the derived class of the estimator.
        Returns
        -------
        name: str
            the name of the estimator constructed dynamically and depended
            of the properties of current estimator
        """
        if hasattr(self, '_model_name'):
            name = self._model_name
        else:
            name = self.__class__.__name__
        if hasattr(self, 'class_weight') and self.class_weight == 'balanced':
            name += 'w'
        if hasattr(self, 'rounddown') and self.rounddown:
            name += 'd'
        if hasattr(self, 'probability') and self.probability:
            name += 'p'
        if hasattr(self, 'pdegree'):
            name += 'P{:d}'.format(self.pdegree)

        if hasattr(self, 'clb') and self.clb != 0:
            if self.clb > 0:
                name += '.iso'
            else:
                name += '.sig'
        return name

    def fit(self, X, y):
        """ Fit meta estimator using train dataset and targets

        Parameters
        ----------
        X: array, shape=(n_samples, n_features)
            train data samples with values of their features
        y: array, shape=(n_samples,))
            targets
        """
        self.PipelineList = []
        if self.use_scaler > 0:
            self.PipelineList.append(
                ("scaler", StandardScaler(with_mean=(self.use_scaler > 1)))
            )
        self._pipeline_append(self.PipelineList)
        self.PipelineList.append(("est", self._get_clf(self.sclf)))

        self.rd = Pipeline(self.PipelineList)

        if hasattr(self, 'rounddown') and self.rounddown:
            X, y, _ = round_down(X, y)

        # TODO fix this hack
        # if clb < 0 use sigmoid (HACK)
        if hasattr(self, 'clb') and self.clb != 0:
            if hasattr(self, 'clb_method'):
                method = self.clb_method
            elif self.clb < 0:
                method = 'sigmoid'
            else:
                method = 'isotonic'
            clb = self.clb if self.clb > 0 else -self.clb
            cv = bootstrap_632(len(y), clb, random_state=self.rs)
            self.rd = CalibratedClassifierCV(self.rd, cv=cv, method=method)

        self.rd.fit(X, y)

        # adjust threshold
        if hasattr(self.rd, 'predict_proba'):
            y_prob = self.rd.predict_proba(X)[:, 1]
            self.threshold, self.threshold2 = best_threshold(y, y_prob)
        else:
            self.threshold = self.threshold2 = None

        return self

    def _pipeline_append(self, pipelineList):
        """ Append any transformers into current pipeline.

        Virtual function: allows derived classes to append any new
        transformers into current pipeline. By default does nothing.

        Parameters
        ----------
        pipelineList: list
            a list that should be added transformers
        """
        pass

    def predict(self, X):
        """Predict the target of new samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples,)
            The predicted class.
        """
        if hasattr(self.rd, "predict_proba") and self.threshold:
            y_prob = self.rd.predict_proba(X)[:, 1]
            return np.asarray(y_prob > self.threshold, dtype=int)
        else:
            return self.rd.predict(X)

    def get_param_grid(self, randomized=True):
        "get param_grid for this model to use with GridSearch"
        return []

    @property
    def coef_(self):
        """
        """
        clf = self.rd.named_steps['est']
        if hasattr(clf, 'coef_'):
            return clf.coef_
        elif hasattr(clf, 'feature_importances_'):
            return clf.feature_importances_
        else:
            raise RuntimeError("est without coef_ of feature_importances_")

    @property
    def clf_(self):
        """
        """
        return self.rd.named_steps['est']


class CModel(Model):
    """ The base class to inherit from it all our meta estimators.

    Parameters
    ----------
    sclf: str, optional (default=None)
        name of the base estimator if we want to use this class (Model)
        as our meta estimator (do not inherit)
    use_scaler: int, optional (default=2)
        to use standard scaler during transformation phase
    rounddown: bool, optional (default=False)
        round down classes in the training dataset before fitting
    n_jobs:int, optional (default=1)
        number of cores to use to speed up calculations
    rs:int, optional (default=random_state)
        random seed to initialize the random generator
        by default will use the global parameter random_state

    Attributes
    ----------
    Examples
    --------
    References
    ----------
    """
    def __init__(self, sclf=None, use_scaler=2, rounddown=False,
                 n_jobs=1, rs=None):
        self.sclf = sclf
        self.use_scaler = use_scaler
        self.rounddown = rounddown
        self.n_jobs = n_jobs
        self.rs = rs

    def predict_proba(self, X):
        """Posterior probabilities of classification

        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            The predicted probas.
        """
        if hasattr(self.rd, "predict_proba"):
            return self.rd.predict_proba(X)
        else:
            raise RuntimeError("rd without predict_proba")

    def _get_clf(self, sclf):
        if sclf is None:
            raise NotImplementedError('virtual function')
        else:
            return get_clf(sclf, n_jobs=self.n_jobs, random_state=self.rs)


class LR2(CModel):
    def __init__(self, C=1.0, class_weight='balanced', rounddown=False,
                 use_scaler=2, clb=0):
        super(LR2, self).__init__(rounddown=rounddown)
        self.C = C
        self.class_weight = class_weight
        self.rounddown = rounddown
        self.use_scaler = use_scaler
        self.clb = clb

    def _get_clf(self, sclf):
        clf = lm.LogisticRegression(
            penalty='l2', dual=False, C=self.C,
            class_weight=self.class_weight, random_state=self.rs,
            solver='lbfgs')
        return clf


class LR1(CModel):
    def __init__(self, C=1.0, class_weight='balanced', rounddown=False,
                 use_scaler=2, clb=0):
        super(LR1, self).__init__(rounddown=rounddown)
        self.C = C
        self.class_weight = class_weight
        self.rounddown = rounddown
        self.use_scaler = use_scaler
        self.clb = clb

    def _get_clf(self, sclf):
        clf = lm.LogisticRegression(
            penalty='l1', dual=False, C=self.C,
            class_weight=self.class_weight, random_state=self.rs,
            solver='liblinear')
        return clf


class LR2CV(LR2):
    def _get_clf(self, sclf):
        clf = lm.LogisticRegressionCV(
            Cs=10, penalty='l2', dual=True,
            scoring='roc_auc', cv=3, n_jobs=1,
            class_weight=self.class_weight, solver='liblinear')
        return clf


class SVCL(CModel):
    """ C-Support Vector Classification.

    The implementation is based on libsvm. The fit time complexity is more
    than quadratic with the number of samples which makes it hard to scale
    to dataset with more than a couple of 10000 samples.

    References
    ----------
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """
    def __init__(self, C=0.001, class_weight='balanced', probability=False,
                 rounddown=False, clb=0):
        super(SVCL, self).__init__(rounddown=rounddown)
        self.C = C
        self.class_weight = class_weight
        self.probability = probability
        self.rounddown = rounddown
        self.clb = clb

    def _get_clf(self, sclf):
        clf = svm.SVC(
            C=self.C, kernel='linear', probability=self.probability,
            class_weight=self.class_weight, verbose=0, random_state=self.rs)
        return clf


class LSVC(CModel):
    """ Linear Support Vector Classification.

    Similar to SVC with parameter kernel=’linear’, but implemented in terms
    of liblinear rather than libsvm, so it has more flexibility in the choice
    of penalties and loss functions and should scale better (to large numbers
    of samples).

    References
    ----------
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    """
    def __init__(self, C=1.0, loss='squared_hinge', penalty='l2', dual=True,
                 class_weight='balanced', rounddown=False, clb=0):
        super(LSVC, self).__init__(rounddown=rounddown)
        self.C = C
        self.loss = loss
        self.penalty = penalty
        self.dual = dual
        self.class_weight = class_weight
        self.rounddown = rounddown
        self.clb = clb

    def _get_clf(self, sclf):
        clf = svm.LinearSVC(C=self.C, loss=self.loss, penalty=self.penalty,
                            dual=self.dual, verbose=0,
                            class_weight=self.class_weight,
                            random_state=self.rs)
        return clf


class SVC(CModel):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=0.0002, coef0=1.0,
                 class_weight='balanced', probability=False, rounddown=False,
                 clb=0, n_jobs=1):
        super(SVC, self).__init__(rounddown=rounddown)
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.class_weight = class_weight
        self.probability = probability
        self.rounddown = rounddown
        self.clb = clb
        self.n_jobs = n_jobs

    @property
    def _model_name(self):
        name = self.__class__.__name__
        if self.kernel == 'rbf':
            name += '.rbf.'
        elif self.kernel == 'poly':
            name += '.poly{:d}.'.format(self.degree)
        else:
            name += "." + self.kernel + "."
        return name

    def _get_clf(self, sclf):
        clf = svm.SVC(C=self.C, kernel=self.kernel, degree=self.degree,
                      gamma=self.gamma, coef0=self.coef0,
                      probability=self.probability, verbose=0,
                      class_weight=self.class_weight)
        return clf


class SVCR(SVC):
    def __init__(self, C=1.0, degree=3, gamma=0.0002, coef0=1.0,
                 class_weight='balanced', probability=False, rounddown=False,
                 clb=0, n_jobs=1):
        super(SVCR, self).__init__(
            C=C, kernel='rbf', degree=degree, gamma=gamma, coef0=coef0,
            class_weight=class_weight, probability=probability,
            rounddown=rounddown, clb=clb, n_jobs=n_jobs)


class SVCP(SVC):
    def __init__(self, C=1.0, degree=3, gamma=0.0001, coef0=1.0,
                 class_weight='balanced', probability=False, rounddown=False,
                 clb=0, n_jobs=1):
        super(SVCP, self).__init__(
            C=C, kernel='poly', degree=degree, gamma=gamma, coef0=coef0,
            class_weight=class_weight, probability=probability,
            rounddown=rounddown, clb=clb, n_jobs=n_jobs)


class SVCRg(CModel):
    def __init__(self, class_weight='balanced', probability=False,
                 rounddown=False, clb=0, n_jobs=1,
                 pgrid_str=None):
        super(SVCRg, self).__init__(rounddown=rounddown)
        self.class_weight = class_weight
        self.probability = probability
        self.rounddown = rounddown
        self.clb = clb
        self.n_jobs = n_jobs
        self.pgrid_str = pgrid_str

    def _get_clf(self, sclf):
        C_range = 10.0 ** np.arange(-3, 4)
        # gamma_range = 10.0 ** np.arange(-4, 3)
        # svm3 = dict(gamma=gamma_range, C=C_range)
        # svm2 = dict(gamma=gamma_range)
        svm1 = dict(C=C_range)
        pg = svm1 if self.pgrid_str is None else param_grids(self.pgrid_str)
        est3 = svm.SVC(C=1.0, kernel='rbf', gamma=0.1,
                       probability=self.probability, verbose=0,
                       class_weight=self.class_weight)
        ncv = 4
        clf = model_selection.GridSearchCV(est3, pg, scoring='roc_auc', cv=ncv,
                                       n_jobs=self.n_jobs, verbose=0)
        return clf


class SVCPg(CModel):
    def __init__(self, class_weight='balanced', probability=False,
                 rounddown=False, clb=0, n_jobs=1, pdegree=2,
                 pgrid_str=None):
        super(SVCPg, self).__init__(rounddown=rounddown)
        self.class_weight = class_weight
        self.probability = probability
        self.rounddown = rounddown
        self.clb = clb
        self.n_jobs = n_jobs
        self.pdegree = pdegree
        self.pgrid_str = pgrid_str

    def _get_clf(self, sclf):
        # svm3 = {'C':[0.001,0.01,0.1,1.0,10],
        # 'gamma':[0.1,0.01,0.001,0.0001], 'coef0':[0,1]}
        # svm2 = {'gamma':[0.1,0.01,0.001,0.0001], 'coef0':[0,1]}
        svm1 = {'C': [0.001, 0.01, 0.1, 1.0, 10, 100], 'coef0': [0, 1]}
        pg = svm1 if self.pgrid_str is None else param_grids(self.pgrid_str)
        est4 = svm.SVC(C=1.0, kernel='poly', gamma=0.1,
                       probability=self.probability, degree=self.pdegree,
                       verbose=0, class_weight=self.class_weight)
        clf = model_selection.GridSearchCV(est4, pg, scoring='roc_auc', cv=4,
                                       n_jobs=self.n_jobs, verbose=0)
        return clf


class RF(CModel):
    def __init__(self,
                 n_estimators=10,
                 min_samples_leaf=1,
                 max_depth=None,
                 max_features='auto',
                 n_jobs=1,
                 class_weight='balanced', rounddown=False,
                 use_scaler=2, clb=0):
        super(RF, self).__init__(rounddown=rounddown)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.rounddown = rounddown
        self.use_scaler = use_scaler
        self.clb = clb

    def _get_clf(self, sclf):
        clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                max_features=self.max_features,
                class_weight=self.class_weight,
                n_jobs=self.n_jobs,
                random_state=self.rs,
                verbose=0)
        return clf


class RFg(RF):
    def _get_clf(self, sclf):
        clf1 = RandomForestClassifier(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                max_features=self.max_features,
                class_weight=self.class_weight,
                n_jobs=1,
                random_state=self.rs,
                verbose=0)
        rf1 = {'max_depth': [2, 4, 8, 16, 24, 32]}
        clf = model_selection.GridSearchCV(clf1, rf1, cv=4, n_jobs=self.n_jobs,
                                       verbose=0)
        return clf


# --- classification metrics --#

def jaccard_binary_index(y_true, y_pred):
    """ Jaccard binary index.

    Jaccard binary index as defined in
    https://en.wikipedia.org/wiki/Jaccard_index

    Parameters
    ----------
    y_true: array, shape = [n_points,]
        True binary class values
    y_pred: array, shape = [n_points,]
        Predicted binary class values
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape
    and_sum = np.logical_and(y_true == 1, y_pred == 1).sum()
    or_sum = np.logical_or(y_true == 1, y_pred == 1).sum()
    return float(and_sum) / or_sum if or_sum > 0 else 0.0


def test_jaccard_binary_index():
    y_test = np.array([0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0])
    sk_jaccard_score = metrics.jaccard_similarity_score(y_test, y_pred)
    print(sk_jaccard_score)
    jaccard_index = jaccard_binary_index(y_test, y_pred)
    print(jaccard_index)
    assert jaccard_index == 0.5

# --- helpers -----------------#


def make_skewed_data(random_state=None):
    X, y = make_classification(
        n_samples=5000, n_features=20, n_classes=2,
        n_clusters_per_class=2, n_informative=8, n_redundant=2,
        random_state=random_state)
    # create unbalamced classes
    plus = np.where(y > 0)[0]
    minus = np.where(y <= 0)[0]
    plus_sel = random.sample(plus, int(len(plus) / 25))
    sel = np.r_[minus, plus_sel]
    np.sort(sel)
    return X[sel, :], y[sel]


def create_df_circles(N=3000, circles=1, p=0.05, p2=0.01, rx=0.03):
    """ Create skewed dataset with circles.
    """
    p = p / circles
    X = np.random.rand(N, 2)
    X = X * 2 - 1.0
    X1 = X + np.random.randn(N, 2) * rx

    if circles == 1:
        y = (X1[:, 0] ** 2 + X1[:, 1] ** 2) < p
        y = np.asarray(y, dtype=int)
    elif circles == 2:
        c1 = (-0.5, -0.5)
        y1 = ((X1[:, 0] + c1[0]) ** 2 + (X1[:, 1] + c1[1]) ** 2) < p
        c2 = (0.2, 0.2)
        y2 = ((X1[:, 0] + c2[0]) ** 2 + (X1[:, 1] + c2[1]) ** 2) < p
        y = np.asarray(y1 | y2, dtype=int)
    elif circles == 3:
        c1 = (-0.5, -0.5)
        y1 = ((X1[:, 0] + c1[0]) ** 2 + (X1[:, 1] + c1[1]) ** 2) < p
        c2 = (0.3, -0.2)
        y2 = ((X1[:, 0] + c2[0]) ** 2 + (X1[:, 1] + c2[1]) ** 2) < p
        c3 = (-0.4, 0.5)
        y3 = ((X1[:, 0] + c3[0]) ** 2 + (X1[:, 1] + c3[1]) ** 2) < p
        y = np.asarray(y1 | y2 | y3, dtype=int)
    else:
        raise ValueError("wrong number of circles: {}".format(circles))

    # add even noise
    noise = (np.random.rand(N) < p2) + 0
    y = (y + noise) % 2
    print("Classes balance:", np.unique(y, return_counts=True))
    return pd.DataFrame(np.c_[X, y], columns=('x', 'y', 'goal'))


def check_model(model, X, y, test_size=0.20, train_stat=False,
                random_state=None):
    """
    """
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)
    print("model: {}".format(model.name))
    # print(model.PipelineList)
    y_pred = model.predict(X_test)
    # print("y_pred:",y_pred)
    y_proba = model.predict_proba(X_test)[:, 1]
    # print("y_proba:", model.predict_proba(X_test))
    target_names = ['class 0', 'class 1']
    y_true = y_test

    def print_stats():
        print(metrics.classification_report(y_true, y_pred,
              target_names=target_names))
        print("roc_auc_score: {:1.4f} | LogLoss: {:1.3f} | Brier score loss:"
              " {:1.3f}".format(metrics.roc_auc_score(y_true, y_proba),
                                metrics.log_loss(y_true, y_proba),
                                metrics.brier_score_loss(y_true, y_proba)))
        if hasattr(model, 'threshold') and model.threshold:
            precision, sensitivity, specificity = \
                precision_sensitivity_specificity(y_true, y_proba,
                                                  threshold=model.threshold)
            print("sensitivity(recall): {:1.2f} and specificity: {:1.2f}"
                  " with threshold={:1.2f}".format(
                      sensitivity, specificity, model.threshold))
    print_stats()
    if train_stat:
        y_pred, y_true = model.predict(X_train), y_train
        y_proba = model.predict_proba(X_train)[:, 1]
        print("train stats:")
        print_stats()

# ------- Base Classifier CModel -----------###


class LinearRegression_proba(lm.LinearRegression):
    def predict_proba(self, X):
        y = self.predict(X)
        y = 1./(1+np.exp(-(y-0.5)))
        return np.vstack((1-y, y)).T


class LassoCV_proba(lm.LassoCV):
    def predict_proba(self, X):
        logger.debug('alpha_=%s', self.alpha_)
        y = self.predict(X)
        y = 1./(1+np.exp(-(y-0.5)))
        return np.vstack((1-y, y)).T


class RidgeCV_proba(lm.RidgeCV):
    def predict_proba(self, X):
        logger.debug('alpha_=%s', self.alpha_)
        y = self.predict(X)
        if 0:
            y_min, y_max = y.min(), y.max()
            if y_max > y_min:
                y = (y-y_min)/(y_max-y_min)
        else:
            y = 1./(1+np.exp(-(y-0.5)))
        return np.vstack((1-y, y)).T


class KNeighborsClassifier_proba(KNeighborsClassifier):
    def predict_proba(self, X):
        y = super(KNeighborsClassifier_proba, self).predict_proba(X)
        y[np.isnan(y)] = 0.5
        return y


class SVC_proba(svm.SVC):
    def predict_proba(self, X):
        if hasattr(self, "decision_function"):  # use decision function
            prob_pos = self.decision_function(X)
            y = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            return np.vstack((1-y, y)).T
        else:
            raise RuntimeError("svm.SVC without decision_function")


class ConstClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, c=0):
        self.c = c

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        y1 = np.empty(X.shape[0])
        y1.fill(self.c)
        y_proba = np.vstack((1-y1, y1)).T
        return y_proba

    def predict(self, X):
        return self.predict_proba(X)[:, 1]


class MeanClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        y1 = np.mean(X, axis=1)
        y_proba = np.vstack((1-y1, y1)).T
        return y_proba

    def predict(self, X):
        return self.predect_proba()[:, 1]


class RoundClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier with rounding classes
    """
    def __init__(self, est, rup=0, find_cutoff=False):
        self.est = est
        self.rup = rup
        self.find_cutoff = find_cutoff

    def fit(self, X, y):
        from imbalanced import (find_best_cutoff, round_smote,
                                round_down, round_up)
        if self.rup > 0:
            X1, y1, _ = round_up(X, y)
        elif self.rup < 0:
            if self.rup < -1:
                X1, y1 = round_smote(X, y)
            else:
                X1, y1, _ = round_down(X, y)
        else:
            X1, y1 = X, y
        self.est.fit(X1, y1)
        if self.find_cutoff:
            ypp = self.predict_proba(X)[:, 1]
            self.cutoff = find_best_cutoff(y, ypp)
        else:
            self.cutoff = 0.5
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        if hasattr(self.est, 'predict_proba'):
            y_proba = self.est.predict_proba(X)
        else:
            y = self.est.predict(X)
            y_proba = np.vstack((1-y, y)).T
        return y_proba

    def predict(self, X):
        if not self.find_cutoff and hasattr(self.est, 'predict'):
            return self.est.predict(X)
        ypp = self.predict_proba(X)[:, 1]
        return np.array(map(int, ypp > self.cutoff))


def test():
    print("tests ok")

if __name__ == '__main__':
    random.seed(1)
    import argparse
    parser = argparse.ArgumentParser(description='Tuner.')
    parser.add_argument('cmd', nargs='?', default='test')
    args = parser.parse_args()
    print(args, file=sys.stderr)

    if args.cmd == 'test':
        test()
    else:
        raise ValueError("bad cmd")
