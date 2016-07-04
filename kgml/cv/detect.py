#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev
# License:  BSD 3 clause
"""
    Object Detector
"""
from __future__ import division, print_function
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import matplotlib.patches as patches
import cPickle as pickle
from util import rgb_to_hsv, hsv_to_rgb


def transform_hs(img, hue_min=0.45, hue_max=0.60, satur_min=0.4):
    """ Transform image using hue and saturation features """
    # select using hue
    hue = rgb_to_hsv(img)[:, :, 0]
    img = img.copy()
    img[hue < hue_min] = 0
    img[hue > hue_max] = 0

    # select usin saturation
    img = rgb_to_hsv(img)[:, :, 1]
    binary_img = img > satur_min

    # Remove small white regions
    open_img = ndimage.binary_opening(binary_img)
    # Remove small black hole
    close_img = ndimage.binary_closing(open_img)

    return close_img


class HueSaturationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bins=31, min1=0.01, min0=0.0, min_ratio=2,
                 min_satur=0.4):
        self.bins = bins
        self.min1 = min1
        self.min0 = min0
        self.min_ratio = min_ratio
        self.min_satur = min_satur

    def get(self):
        data = (self.bins, self.min1, self.min0,
                self.min_ratio, self.min_satur,
                self.bin_edges, self.sels,
                self.hist_1, self.hist_0)
        return data

    def set(self, data):
        (self.bins, self.min1, self.min0,
         self.min_ratio, self.min_satur,
         self.bin_edges, self.sels,
         self.hist_1, self.hist_0) = data

    def dump(self, filepath):
        with open(filepath, 'wb') as fp:
            pickle.dump(self.get(), fp)

    def load(self, filepath):
        with open(filepath, 'rb') as fp:
            data = pickle.load(fp)
        self.set(data)

    def fit(self, X, Y=None):
        assert Y is not None
        assert X.shape == Y.shape
        X = rgb_to_hsv(X)
        X = X.reshape((-1, 3))[:, 0]
        y = Y.reshape((-1, 3))[:, 0]

        X_pool = X[y.astype(bool)]
        hist_1, bin_edges_1 = np.histogram(X_pool, range=(0, 1),
                                           bins=self.bins,
                                           density=True)
        hist_1 /= self.bins
        X_nopool = X[np.logical_not(y.astype(bool))]
        hist_0, bin_edges_0 = np.histogram(X_nopool, range=(0, 1),
                                           bins=self.bins,
                                           density=True)
        hist_0 /= self.bins
        assert tuple(bin_edges_1) == tuple(bin_edges_0)
        # print(len(bin_edges_1), len(hist_1))
        # print(zip(bin_edges_1, hist_0, hist_1))

        self.bin_edges = bin_edges_1
        self.hist_1 = hist_1
        self.hist_0 = hist_0
        n_hist = len(hist_1)
        assert len(hist_0) == n_hist
        sels = np.zeros(n_hist, dtype=int)
        for i in range(n_hist):
            if hist_1[i] > self.min1:
                if hist_0[i] <= self.min0:
                    sels[i] = 1
                elif hist_1[i] / hist_0[i] > self.min_ratio:
                    sels[i] = 1
        self.sels = sels
        # print(zip(bin_edges_1, hist_0, hist_1, sels))

        return self

    def plot_histograms(self, figsize=(10, 5), bars=True):
        fig, ax = plt.subplots(figsize=figsize)
        if bars:
            idx = np.arange(len(self.hist_1))
            width = 0.33
            ax.bar(idx+width, self.hist_1, width, color='blue', label='pool')
            ax.bar(idx+2*width, self.hist_0, width, color='red',
                   label='nopool')
            ax.bar(idx, self.sels, width, color='cyan', label='sel')
        else:
            bin_centers = 0.5*(self.bin_edges[:-1] + self.bin_edges[1:])
            ax.plot(bin_centers, self.hist_1, lw=2, color='blue',
                    label='pool')
            ax.plot(bin_centers, self.hist_0, lw=2, color='red',
                    label='nopool')
            ax.set_xlim((0, 1))
        ax.set_ylim((0, max(self.hist_1.max(), self.hist_0.max())))
        plt.legend()
        plt.show()

    def transform_hs(self, X):
        # select using hue
        img = X
        hue = rgb_to_hsv(img)[:, :, 0]
        img = img.copy()

        edges = self.bin_edges
        for i, sel in enumerate(self.sels):
            if sel == 0:
                s = np.logical_and(hue > edges[i], hue < edges[i+1])
                img[s] = 0

        # select usin saturation
        img = rgb_to_hsv(img)[:, :, 1]
        binary_img = img > self.min_satur

        # Remove small white regions
        open_img = ndimage.binary_opening(binary_img)
        # Remove small black hole
        close_img = ndimage.binary_closing(open_img)

        return close_img

    def transform(self, X):
        return self.transform_hs(X)


def create_test_image(x1=10, y1=20, d=10, c=0.55):
    """Create test image with blue rect inside."""
    shape = (32, 64, 3)
    img = np.full(shape, 0.5)
    img[:, :30, 0] = 0.10
    img[:, 30:, 0] = 0.90
    img[y1:y1+d, x1:x1+d, 0] = c
    img = hsv_to_rgb(img)
    return img


def test_transform_hs():
    """Create image and transform it."""
    img = create_test_image()
    fig, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img, cmap='gray')
    img2 = transform_hs(img)
    axarr[1].imshow(img2, cmap='gray')
    plt.show()


def show_patch(ax, p, color='black'):
    """ Show patch using annotation. """
    ax.add_patch(
        patches.Rectangle(
            (p[0], p[1]),   # (x,y)
            p[2],          # width
            p[2],          # height
            hatch='\\',
            fill=False,
            color=color)
        )


def zest_HueSaturationTransformer(c=0.35):
    """Create image and transform it."""
    p1 = (10, 10, 10)
    p_train = (8, 8, 14)
    p2 = (40, 10, 10)
    img_train = create_test_image(x1=p1[0], y1=p1[1], d=p1[2], c=c)
    img_test = create_test_image(x1=p2[0], y1=p2[1], d=p2[2], c=c)

    Y = np.zeros(img_train.shape)
    x1, x2 = p_train[0], p_train[0] + p_train[2]
    y1, y2 = p_train[1], p_train[1] + p_train[2]
    Y[y1:y2+1, x1:x2+1] = 1
    hst = HueSaturationTransformer()
    hst.fit(img_train, Y)
    img_pred = hst.transform(img_test)

    # hst.plot_histograms()
    # return

    fig, axarr = plt.subplots(4, 1)
    ax = axarr[0]
    ax.imshow(img_train, cmap='gray')
    show_patch(ax, p_train)
    axarr[1].imshow(img_test, cmap='gray')
    img_hs = transform_hs(img_test)
    axarr[2].imshow(img_hs, cmap='gray')
    axarr[3].imshow(img_pred, cmap='gray')
    plt.show()


def test_HueSaturationTransformer():
    zest_HueSaturationTransformer(c=0.55)


def test_HueSaturationTransformer2():
    zest_HueSaturationTransformer(c=0.35)

if __name__ == '__main__':
    # test_transform_hs()
    # test_HueSaturationTransformer()
    test_HueSaturationTransformer2()
