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
from sklearn.base import BaseEstimator, RegressorMixin
from scipy import optimize
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SES(BaseEstimator, RegressorMixin):
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
        self.y_p, self.rmse = self.calculate(self.ts.values, self.alpha)
        self.ts_fit = pd.Series(self.y_p, index=self.ts.index)

    def error_func(self, ts):
        return lambda alpha: self.calculate(ts.values, alpha)[1]

    def plot_error_func(self, ts, ax1=None):
        func = self.error_func(ts)
        vfunc = np.vectorize(func)
        xvec = np.linspace(0, 1, 50)
        yvec = vfunc(xvec)
        x_pred = self.fit_params(ts)
        ax = plt.subplots()[1] if ax1 is None else ax1
        ax.plot(xvec, yvec, label='alpha')
        ax.scatter(x_pred, vfunc(x_pred), c='r',)
        ax.legend(loc='best')
        ax.set_title('Error function')

    def predict(self, horiz=7):
        ts = self.ts.copy()
        y_next = self.alpha * self.y[-1] + (1.0 - self.alpha) * self.y_p[-1]
        for h in range(1, horiz+1):
            goal_datetime = ts.index[-1] + 1
            ts[goal_datetime] = y_next
        return ts.iloc[-horiz:]


def make_timeseries(n=50, h=7, a=2, b=0.03, r=0.5, nan_len=10, rs=None):
    if rs is not None:
        np.random.seed(rs)
    x = np.linspace(0, (np.pi)*n, n*h)
    y = np.sin(x) + a * np.cos(x/h/3) + b + r * np.random.randn(len(x))
    if nan_len > 0:
        nan_start = len(y) // 3
        y[nan_start:nan_start+nan_len] = np.nan
    rng = pd.date_range('1/1/2000', periods=len(y), freq='D')
    # rng = np.arange(len(y))
    return pd.Series(y, index=rng)


def simulate_forecasts(est, ts=None, **params):
    if ts is None:
        ts = make_timeseries(**params)
    if hasattr(est, 'plot_error_func'):
        fig, axarr = plt.subplots(1, 2, figsize=(14, 5))
        est.plot_error_func(ts, axarr[0])
        ax = axarr[1]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    n_train = ts.shape[0] * 4 // 5
    n_test = len(ts) - n_train
    ts_train = ts.iloc[:n_train]
    est.fit(ts_train)
    ts_pred = est.predict(n_test)
    ax.plot(ts, label='data', color='k', alpha=0.5)
    if hasattr(est, 'ts_fit'):
        ax.plot(est.ts_fit, label='train', color='g')
    ax.plot(ts_pred, label='pred', color='r')
    ax.legend(loc='lower right')
    ax.set_title('Forecast')

    import matplotlib.dates as mdates
    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    monthsFmt = mdates.DateFormatter('%b')
    yearsFmt = mdates.DateFormatter('\n%Y')

    ax.xaxis.set_minor_locator(months)
    ax.xaxis.set_minor_formatter(monthsFmt)
    # plt.setp(ax.xaxis.get_minorticklabels(), rotation=90)
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)

    plt.show()

    return ts


def test_SES():
    if __name__ != '__main__':
        plt.ion()
    simulate_forecasts(SES(), rs=1)


if __name__ == '__main__':
    test_SES()
