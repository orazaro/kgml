#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev
# License:  BSD 3 clause
"""
    Adaprive regression of time series.
"""
from __future__ import (division, print_function)
import numpy as np
import pandas as pd
import logging
from scipy import optimize
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SES(object):
    """ Simple exponential smoothing.
    """
    def __init__(self, alpha='auto'):
        self.alpha = alpha

    @staticmethod
    def calculate(y, alpha):
        n = len(y)
        y_p = np.zeros(n)
        y_p[0] = y[0]
        for i in range(1, n):
            if np.isnan(y_p[i-1]):
                y_p[i] = y[i-1]
            else:
                y_p[i] = alpha * y[i-1] + \
                         (1.0 - alpha) * y_p[i-1]
        y_p = np.asarray(y_p)
        rmse = np.sqrt(np.nanmean((y - y_p)**2))
        return y_p, rmse

    def fit_params(self, ts):
        func = self.error_func(ts)
        r = optimize.differential_evolution(func, [(0.0, 1.0)])
        assert r.success
        return r.x[0]

    def fit(self, ts):
        if isinstance(self.alpha, basestring):
            if self.alpha == 'auto':
                self.alpha = self.fit_params(ts)
            else:
                raise ValueError("bad alpha: {}".format(self.alpha))
        self.ts = ts
        self.y = self.ts.values
        y_p, rmse = self.calculate(self.ts.values, self.alpha)
        self.y_p = y_p

    def error_func(self, ts):
        return lambda alpha: self.calculate(ts.values, alpha)[1]

    def plot_error_func(self, ts, ax1=None):
        func = self.error_func(ts)
        vfunc = np.vectorize(func)
        xvec = np.linspace(0, 1, 50)
        yvec = vfunc(xvec)
        x_pred = self.fit_params(ts)
        ax = plt.subplots()[1] if ax1 is None else ax1
        ax.plot(xvec, yvec)
        ax.scatter(x_pred, vfunc(x_pred), c='r',)

    def predict(self, horiz=7):
        ts = self.ts.copy()
        y_next = self.alpha * self.y[-1] + (1.0 - self.alpha) * self.y_p[-1]
        for h in range(1, horiz+1):
            goal_datetime = ts.index[-1] + 1
            ts[goal_datetime] = y_next
        return ts[-horiz:]


def test_SES():
    if __name__ != '__main__':
        plt.ion()
    x = np.linspace(0, np.pi*20, 200)
    y = np.sin(x) + 0.0*x + 0.3 + np.random.randn(len(x))*0.5
    ts = pd.Series(y, index=x)

    fig, axarr = plt.subplots(1, 2)

    ses = SES()
    ses.plot_error_func(ts, axarr[0])
    axarr[1].plot(ts)
    plt.show()

if __name__ == '__main__':
    test_SES()
