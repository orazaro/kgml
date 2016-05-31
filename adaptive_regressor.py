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
    def __init__(self, alpha='auto', horiz=1):
        self.alpha = alpha
        self.horiz = horiz

    def calculate(self, y, alpha):
        n = len(y)
        level = np.zeros(n)
        level[0] = y[0]
        se = []
        for i in range(1, n):
            # update
            if np.isnan(level[i-1]):
                level[i] = y[i]
            else:
                level[i] = alpha * y[i] + \
                         (1.0 - alpha) * level[i-1]
            # predict and calc error
            if i >= self.horiz and not np.isnan(level[i]):
                for h in range(self.horiz):
                    j = i + 1 + h
                    if j >= n:
                        break
                    if not np.isnan(y[j]):
                        y_p = level[i]
                        se.append((y[j] - y_p)**2)
        level = np.asarray(level)
        rmse = np.sqrt(np.nanmean(se))
        return level, rmse

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

    def predict(self, horiz=7):
        ts = None
        y_next = self.y_p[-1]
        for h in range(1, horiz+1):
            if ts is None:
                goal_datetime = self.ts.index[-1] + 1
                if hasattr(self.ts.index, 'freq'):
                    rng = pd.date_range(goal_datetime, periods=1,
                                        freq=self.ts.index.freq)
                else:
                    rng = np.array([goal_datetime])
                ts = pd.Series([y_next], index=rng)
            else:
                goal_datetime = ts.index[-1] + 1
            ts[goal_datetime] = y_next
        return ts

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


class ExpSmoothing(BaseEstimator, RegressorMixin):
    """ Exponential smoothing with the trend and seasonal components.
    """
    def __init__(self, alpha='auto', beta='auto', gamma='auto', ms=7, horiz=7):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ms = ms
        self.horiz = horiz

    def calculate(self, y, alpha, beta, gamma, ms):
        n = len(y)
        level = np.zeros(n)
        trend = np.zeros(n)
        seas = dict()
        level[0] = np.nanmean(y[:ms])
        trend[0] = np.nanmean([(y[i+ms] - y[i]) / ms for i in range(ms)])
        for i in range(ms):
            seas[-i] = y[ms-i] - level[0]
        se = []
        for i in range(1, n):
            # update level
            x1 = y[i] - seas[i-ms]
            x2 = level[i-1] + trend[i-1]
            level[i] = alpha * x1 + (1.0 - alpha) * x2
            # update trend
            x1 = level[i] - level[i-1]
            trend[i] = beta * x1 + (1.0 - beta) * trend[i-1]
            # update seas
            x1 = y[i] - level[i-1] - trend[i-1]
            seas[i] = gamma * x1 + (1.0 - gamma) * seas[i-ms]
            # predict and calc error
            if i < ms:
                continue
            for h in range(1, ms+1):
                hm = (h - 1) % ms + 1
                j = i + h
                if j < n and not np.isnan(y[j]):
                    y_p = level[i] + h * trend[i] + seas[i-ms+hm]
                    se.append((y[j] - y_p)**2)
        rmse = np.sqrt(np.nanmean(se))
        print("rmse:", rmse)
        return level, trend, seas, rmse

    def fit_params(self, ts):
        func = self.error_func(ts)
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        if False:
            x0 = [1.0, 0, 0]
            r = optimize.minimize(func, x0)
            if not r.success:
                r = optimize.differential_evolution(func, bounds)
        else:
            r = optimize.differential_evolution(func, bounds)
        assert r.success
        return r.x[0], r.x[1], r.x[2]

    def fit(self, ts):
        if isinstance(self.alpha, basestring):
            if self.alpha == 'auto':
                self.alpha, self.beta, self.gamma = self.fit_params(ts)
                # print(self.alpha, self.beta, self.gamma)
                # print(self.calculate(ts.values, self.alpha, self.beta,
                #                     self.gamma, self.ms)[-1])
            else:
                raise ValueError("bad alpha: {}".format(self.alpha))
        self.ts = ts
        self.y = self.ts.values
        self.level, self.trend, self.seas, self.rmse = \
            self.calculate(self.ts.values, self.alpha, self.beta,
                           self.gamma, self.ms)
        y_pred = np.zeros(len(self.y))
        y_pred[0] = self.y[0]
        for i in range(1, len(self.y)):
                y_pred[i] = self.level[i-1] + self.trend[i-1] + \
                    self.seas[i-1+1]
        self.ts_fit = pd.Series(y_pred, index=self.ts.index)

    def predict(self, horiz=7):
        ts = None
        for h in range(1, horiz+1):
            t = len(self.level) - 1
            hm = (h - 1) % self.ms + 1
            y_next = self.level[t-1] + h * self.trend[t-1] + \
                self.seas[t - self.ms + hm]
            if ts is None:
                goal_datetime = self.ts.index[-1] + 1
                if hasattr(self.ts.index, 'freq'):
                    rng = pd.date_range(goal_datetime, periods=1,
                                        freq=self.ts.index.freq)
                else:
                    rng = np.array([goal_datetime])
                ts = pd.Series([y_next], index=rng)
            else:
                goal_datetime = ts.index[-1] + 1
            ts[goal_datetime] = y_next
        # print(ts)
        return ts

    def error_func(self, ts):
        return lambda x: self.calculate(
            ts.values, x[0], x[1], x[2], self.ms)[-1]


def make_timeseries(n=50, h=7, a=2, b=10, r=0.5, nan_len=10, rs=None):
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

    if hasattr(ts.index, 'freq'):
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


def test_ExpSmoothing():
    if __name__ != '__main__':
        plt.ion()
    simulate_forecasts(ExpSmoothing(), rs=1)


if __name__ == '__main__':
    # test_SES()
    test_ExpSmoothing()
