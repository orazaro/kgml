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
from scipy import interpolate
import calendar


def datetime_timestamp(d):
    """ Convert datetime d into POSIX timestamp as float.
    """
    return calendar.timegm(d.timetuple())


def normalize_signal_merge(signal, opt_step=10, roundon=True):
    """ Normalize signal using merge.

    Normalize signal: convert from irregular time-spacing signal to
    semi-equidistant. Merge neighboring points to get an optimal step.

    Parameters
    ----------
    signal: list or array (n_points, 2)
        Points of the signal as pairs of (Time, Amplitude)
    opt_step: float, optional (default=10)
        optimal step between points

    Returns
    -------
    new_signal: list or array (n_points, 2)
        Points of the signal as pairs of (Time, Amplitude)
    """
    def merge_group(group):
        res = np.mean(group, axis=0)
        if roundon:
            res = np.round(res)
        return res
    opt_step -= 1
    new_signal = []
    group = []
    for cur in signal:
        if not group:
            group.append(cur)
            continue
        dx1 = group[-1][0] - group[0][0]
        if dx1 >= opt_step:
            new_signal.append(merge_group(group))
            group = [cur]
        else:
            dx2 = cur[0] - group[0][0]
            if dx2 < opt_step:
                group.append(cur)
            elif dx2 - opt_step > opt_step - dx1:
                new_signal.append(merge_group(group))
                group = [cur]
            else:
                group.append(cur)
    if group:
        new_signal.append(merge_group(group))
    return np.array(new_signal)


def normalize_signal_interpolate(signal, opt_step=10, kind='slinear'):
    """ Normalize signal using interpolate.

    Normalize signal: convert from irregular time-spacing signal to
    equidistant using interpolate

    Parameters
    ----------
    signal: list or array (n_points, 2)
        Points of the signal as pairs of (Time, Amplitude)
    opt_step: float, optional (default=10)
        optimal step between points

    Returns
    -------
    new_signal: list or array (n_points, 2)
        Points of the signal as pairs of (Time, Amplitude)
    """
    x, y = zip(*signal)
    f = interpolate.interp1d(x, y, kind='slinear')

    x_min = np.round(x[0] / opt_step) * opt_step
    if x_min < x[0]:
        x_min += opt_step
    x_max = np.round(x[-1] / opt_step) * opt_step
    if x_max > x[-1]:
        x_max -= opt_step

    # print(x_min, x[0], x[-1], x_max)

    x_new = np.arange(x_min, x_max + opt_step, opt_step)
    y_new = f(x_new)

    return np.c_[x_new, y_new]


def running_mean(x, N=5):
    """
    from http://stackoverflow.com/questions/13728392/\
    moving-average-or-running-mean
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def smooth_convolve(x, window_len=11, window='hanning'):
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


def smooth_running_mean(x, N=5):
    """
    https://habrahabr.ru/post/134375/
    """
    assert N % 2 == 1
    w = np.ones(N, 'd')
    y = np.convolve(w/w.sum(), x, mode='same')
    # fix edges
    n_len = len(y)
    hw = int((N - 1) / 2)
    for i in range(0, hw):
        k1 = 0
        k2 = 2 * i + 1
        # print(k1, k2, x[k1:k2].sum())
        y[i] = x[k1:k2].sum() / k2
        # print(n_len-k2, n_len, x[n_len-k2:n_len].sum())
        y[n_len-1-i] = x[n_len-k2:n_len].sum() / k2
    return y


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
