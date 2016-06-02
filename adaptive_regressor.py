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
import numbers
from sklearn.base import BaseEstimator, RegressorMixin
from scipy import optimize
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ExpSmoothing(BaseEstimator, RegressorMixin):
    """ Exponential smoothing with the trend and seasonal components.
    """
    def __init__(self, alpha='auto', beta='auto', gamma='auto', phi='auto',
                 ms=7, horiz=7):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phi = phi
        self.ms = ms
        self.horiz = horiz

    def calculate(self, y, alpha, beta, gamma, phi, ms):
        n = len(y)
        level = np.zeros(n)
        trend = np.zeros(n)
        seas = dict()
        level[0] = np.nanmean(y[:ms])
        trend[0] = np.nanmean([(y[i+ms] - y[i]) / ms for i in range(ms)])
        for i in range(ms):
            if np.isnan(y[ms-i]):
                seas[-i] = 0.0
            else:
                seas[-i] = y[ms-i] - level[0]
        se = []
        for i in range(1, n):
            # update level
            x2 = level[i-1] + phi * trend[i-1]
            if np.isnan(y[i]):
                level[i] = x2
            else:
                x1 = y[i] - seas[i-ms]
                level[i] = alpha * x1 + (1.0 - alpha) * x2
            # update trend
            x1 = level[i] - level[i-1]
            trend[i] = beta * x1 + (1.0 - beta) * phi * trend[i-1]
            # update seas
            if np.isnan(y[i]):
                seas[i] = seas[i-ms]
            else:
                x1 = y[i] - level[i-1] - phi * trend[i-1]
                seas[i] = gamma * x1 + (1.0 - gamma) * seas[i-ms]
            # predict and calc error
            if i > ms:
                phi_h = 0.0
                for h in range(1, self.horiz+1):
                    hm = (h - 1) % ms + 1
                    phi_h += phi**h
                    j = i + h
                    if j < n and not np.isnan(y[j]):
                        y_p = level[i] + phi_h * trend[i] + seas[i-ms+hm]
                        se.append((y[j] - y_p)**2)
        rmse = np.sqrt(np.nanmean(se))
        # print("rmse:", rmse)
        return level, trend, seas, rmse

    def fit_params(self, ts):
        if isinstance(self.phi, numbers.Number):
            n_bounds = 3
        else:
            n_bounds = 4
        func = self.error_func(ts)

        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        r = optimize.differential_evolution(func, bounds[:n_bounds])
        assert r.success

        if isinstance(self.phi, numbers.Number):
            return list(r.x) + [self.phi]
        else:
            return list(r.x)

    def fit(self, ts):
        if not isinstance(self.alpha, numbers.Number):
            if self.alpha == 'auto':
                self.alpha, self.beta, self.gamma, self.phi = \
                    self.fit_params(ts)
                # print(self.alpha, self.beta, self.gamma)
                # print(self.calculate(ts.values, self.alpha, self.beta,
                #                     self.gamma, self.ms)[-1])
            else:
                raise ValueError("bad alpha: {}".format(self.alpha))
        self.ts = ts
        self.y = self.ts.values
        self.level, self.trend, self.seas, self.rmse = \
            self.calculate(self.ts.values, self.alpha, self.beta,
                           self.gamma, self.phi, self.ms)
        y_pred = np.zeros(len(self.y))
        y_pred[0] = self.y[0]
        for i in range(1, len(self.y)):
                y_pred[i] = self.level[i-1] + self.trend[i-1] + \
                    self.seas[i-1+1]
        self.ts_fit = pd.Series(y_pred, index=self.ts.index)

    def predict(self, horiz=7):
        ts = None
        phi_h = 0.0
        for h in range(1, horiz+1):
            t = len(self.level) - 1
            hm = (h - 1) % self.ms + 1
            phi_h += self.phi**h
            y_next = self.level[t-1] + phi_h * self.trend[t-1] + \
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
        if isinstance(self.phi, numbers.Number):
            return lambda x: self.calculate(
                ts.values, x[0], x[1], x[2], self.phi, self.ms)[-1]
        else:
            return lambda x: self.calculate(
                ts.values, x[0], x[1], x[2], x[3], self.ms)[-1]

    def plot_decomposition(self, ts=None, figsize=(12, 8)):
        if ts is not None:
            self.fit(ts)
        seas = [self.seas[i] for i in range(len(self.ts))]
        data = np.c_[self.ts, self.level, self.trend, seas]
        assert data.shape[1] == 4
        columns = ['original', 'level', 'slope', 'season']
        df = pd.DataFrame(data, columns=columns, index=self.ts.index)
        title_fmt = "Decomposition: alpha={:.2f} beta={:.2f} gamma={:.2f}"\
                    " phi={:.2f}"
        title = title_fmt.format(
                self.alpha, self.beta, self.gamma, self.phi)
        df.plot(subplots=True, figsize=figsize, title=title)


def make_timeseries(n=30, h=7, a=2, b=10, r=0.5, nan_len=10, rs=None):
    if rs is not None:
        np.random.seed(rs)
    x = np.linspace(0, (2*np.pi)*n, n*h)
    y = np.sin(x) + a * np.cos(x/h/5) + b + r * np.random.randn(len(x))
    if nan_len > 0:
        nan_start = len(y) // 3
        y[nan_start:nan_start+nan_len] = np.nan
    rng = pd.date_range('1/1/2000', periods=len(y), freq='D')
    # rng = np.arange(len(y))
    return pd.Series(y, index=rng)


def plot_forecasts(est, ts, test_size=0.2, figsize=(12, 5),
                   title='Forecast', **params):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    n_train = int(ts.shape[0] * (1 - test_size))
    n_test = len(ts) - n_train
    ts_train = ts.iloc[:n_train]
    est.fit(ts_train)
    ts_pred = est.predict(n_test)
    ax.plot(ts, label='data', color='k', alpha=0.5)
    if hasattr(est, 'ts_fit'):
        ax.plot(est.ts_fit, label='train', color='g')
    ax.plot(ts_pred, label='pred', color='r')
    ax.legend(loc='lower right')
    ax.set_title(title)

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


def simulate_forecasts(est, test_size=0.2, figsize=(12, 5),
                       title='Simulate and forecast',
                       **params):
    """ Simulate timeseries and plot forecasts. """
    ts = make_timeseries(**params)
    plot_forecasts(est, ts, test_size=test_size, figsize=figsize, title=title,
                   **params)
    return ts


def test_ExpSmoothing():
    if __name__ != '__main__':
        plt.ion()
    simulate_forecasts(ExpSmoothing(phi=1), nan_len=10, rs=1)


def test_ExpSmoothing2():
    ts = make_timeseries(n=50, rs=1)
    est = ExpSmoothing()
    est.fit(ts)
    ts_pred = est.predict()
    print(ts_pred)


if __name__ == '__main__':
    # test_ExpSmoothing()
    test_ExpSmoothing2()
