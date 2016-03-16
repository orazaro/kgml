#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev
# License:  BSD 3 clause
"""
    Time Series
"""
from __future__ import division, print_function

import numpy as np
import statsmodels.api as sm
from pandas.tools.plotting import lag_plot, autocorrelation_plot


def running_mean(x, N=5):
    """
    from http://stackoverflow.com/questions/13728392/\
    moving-average-or-running-mean
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an
                    odd integer
        window: the type of window from 'flat', 'hanning', 'hamming',
                'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
    numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead
          of a string
    NOTE: length(output) != length(input), to correct this:
          return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', \
                'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    # return y
    return y[(int(window_len/2)):-(int(window_len/2))]


def calc_jarque_bera(ts_data, verbosity=1):
    from statsmodels.iolib.table import SimpleTable
    row = [u'JB', u'p-value', u'skew', u'kurtosis']
    jb_test = sm.stats.stattools.jarque_bera(ts_data)
    if verbosity > 0:
        a = np.vstack([jb_test])
        itog = SimpleTable(a, row)
        print(itog)
    if jb_test[1] > 0.05:
        print("the distribution is normal")
    else:
        print("the distribution isn't normal")
    return jb_test


def calc_adfuller(ts_data, verbosity=1):
    test = sm.tsa.adfuller(ts_data)
    print('adf: ', test[0])
    print('p-value: ', test[1])
    print('Critical values: ', test[4])
    if test[0] > test[4]['5%']:
        print("there are unit roots, a time series are not stationary")
    else:
        print("there aren't unit roots, a time series are stationary")


def check_normality_stationarity(ts_data, verbosity=1):
    calc_jarque_bera(ts_data, verbosity=verbosity)
    calc_adfuller(ts_data, verbosity=verbosity)


def ts_plots(rets, figsize=(12, 10)):
    import matplotlib.pyplot as plt
    fig, axarr = plt.subplots(2, 2, sharex=False, sharey=False,
                              figsize=figsize)
    axgen = (e for e in np.array(axarr).ravel())

    rets.plot(kind='line', ax=axgen.next())  # .set_title("data")
    rets.plot(kind='hist', bins=50, ax=axgen.next())  # .set_title("histogram")
    # rets.plot(kind='density',ax=axgen.next()).set_title("density")
    lag_plot(rets, lag=1, ax=axgen.next())  # .set_title("")
    autocorrelation_plot(rets, ax=axgen.next())
    # ax.set_title("autocorrelation plot")


def plot_autocorr(series, lags=25):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(series.squeeze(), lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(series, lags=25, ax=ax2)


def plot_autocorr2(series, lags=30, ylim=(-0.2, 0.2), figsize=(12, 4)):
    import matplotlib.pyplot as plt
    fig, axarr = plt.subplots(1, 2, sharex=False, sharey=False,
                              figsize=figsize)
    axgen = (e for e in np.array(axarr).ravel())
    ax = axgen.next()
    sm.graphics.tsa.plot_acf(series, lags=lags, ax=ax)
    ax.set_title("Autocorrelation")
    ax.set_ylim(ylim)
    ax = axgen.next()
    sm.graphics.tsa.plot_pacf(series, lags=lags, ax=ax)
    ax.set_title("Partial Autocorrelation")
    ax.set_ylim(ylim)
