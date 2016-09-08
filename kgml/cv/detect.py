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
import matplotlib.image as mpimg
from scipy import ndimage
import matplotlib.patches as patches
import cPickle as pickle
from util import rgb_to_hsv, hsv_to_rgb, img_norm
from features import rgb2gray


def transform_hs(img, hue_min=0.45, hue_max=0.60, satur_min=0.4,
                 gray_min=None):
    """ Transform image using hue and saturation features.

    Parameters
    ----------
    img: np.array shape (W, H, 3)
        RGB image as a numpy array
    hue_min: float, optional (default=0.45)
        min value of hue to cut image by hue values
    hue_max: float, optional (default=0.60)
        max value of hue to cut image by hue values
    satur_min: float, optional (default=0.4)
        min value of hue to cut image by saturation values
        (after rgb-hue transform and hue cut)
    gray_min: float, optional (default=None)
        min value to cut 1D image by amplitude
        (after rgb-hue transform, hue cut and 3D->1D gray transform)
        if is None, than don't use the last stage

    Returns
    -------
    img: np.array shape (W, H, 1)
        final binary image
    """
    # select using hue
    hue = rgb_to_hsv(img)[:, :, 0]
    img = img.copy()
    img[hue < hue_min] = 0
    img[hue > hue_max] = 0

    sat = rgb_to_hsv(img)[:, :, 1]
    if gray_min is None:
        # select using saturation
        binary_img = sat > satur_min
    else:
        img[sat < satur_min] = 0
        img = img_norm(img)
        img = rgb2gray(img)
        if gray_min <= 0:
            return img
        binary_img = img > gray_min

    # Remove small white regions
    open_img = ndimage.binary_opening(binary_img)
    # Remove small black hole
    close_img = ndimage.binary_closing(open_img)

    return close_img


class HueSaturationTransformer(BaseEstimator, TransformerMixin):
    """ Transform image using hue and saturation features.

    Parameters
    ----------
    bins: int, optional (default=31)
        number of bins in the hue histogram (after rgb-hst transform)
    min1: float, optional (default=0.01)
        minimal 'plus' density to use bin in the cut
    min0: float, optional (default=0.0)
        maximal 'minus' density to use bin in the cut automatically
    min_ratio: float, optional (default=2.0)
        minimal 'plus' / 'minus' ration to use bin in the cut
    min_satur: float, optional (default=0.4)
        min value of hue to cut (or binarize) image by saturation values
        (after rgb-hue transform and hue cut)
    min_gray: float, optional (default=None)
        min value to binarize 1D image by amplitude
        (after rgb-hue transform, hue cut and 3D->1D gray transform)
        if is None, than don't use the last stage
        elif min_gray=0, than don't binarize the transformed image and
            return 1D gray one
        else cut image with min_satur and then binarize it with min_gray
    """
    def __init__(self, bins=31, min1=0.01, min0=0.0, min_ratio=2.0,
                 min_satur=0.4, min_gray=None):
        self.bins = bins
        self.min1 = min1
        self.min0 = min0
        self.min_ratio = min_ratio
        self.min_satur = min_satur
        self.min_gray = min_gray

    @property
    def attributes(self):
        """
        Returns
        -------
        list
            list of the object attributes to save on object dump
        """
        return ('bins', 'min1', 'min0',
                'min_ratio', 'min_satur', 'min_gray',
                'bin_edges', 'sels',
                'hist_1', 'hist_0')

    def get(self):
        """ Get object attribute as a dict to dump object. """
        data = {}
        for sattr in self.attributes:
            if hasattr(self, sattr):
                data[sattr] = getattr(self, sattr)
        return data

    def set(self, data):
        """ Set the object attributes from the dict 'data'. """
        for k in data:
            setattr(self, k, data[k])

    def dump(self, filepath):
        """ Dump the object to the file. """
        with open(filepath, 'wb') as fp:
            pickle.dump(self.get(), fp)

    def load(self, filepath):
        """ Load the object from the file. """
        with open(filepath, 'rb') as fp:
            data = pickle.load(fp)
        self.set(data)

    def fit(self, X, Y=None):
        """ Fit transformer using images and marks.

        Parameters
        ----------
        X: np.array of shape (W*N, H, 3)
            N RGB images concateneted into one
        Y: np.array of shape (W*N, H, 3)
            Pixel marks of all images from X (0 or 1 values)

        Returns
        -------
        self: the object
        """
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
        """ Plot histograms of 'plus' pixels and 'minus' pixels
        """
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

    def transform_hs(self, X, binarize=True):
        """ Transform RGB image using hue and saturation.

        Parameters
        ----------
        X: np.array of float, shape (W, H, 3)
            RGB image to be transformed
        binarize: bool, optional (default=True)
            to binarize the output image or not

        Returns
        -------
        img: np.array of float, shape (W, H)
            Binary 1D image after transformation if binarize is True
            Gray 1D image after transformation if binarize is False
        """
        # select using hue
        img = X
        hue = rgb_to_hsv(img)[:, :, 0]
        img = img.copy()

        edges = self.bin_edges
        for i, sel in enumerate(self.sels):
            if sel == 0:
                s = np.logical_and(hue > edges[i], hue < edges[i+1])
                img[s] = 0

        sat = rgb_to_hsv(img)[:, :, 1]
        if self.min_gray is None:
            if not binarize:
                return sat
            # select using saturation
            binary_img = sat > self.min_satur
        else:
            img[sat < self.min_satur] = 0
            img = img_norm(img)
            img = rgb2gray(img)
            if not binarize or self.min_gray == 0:
                return img
            binary_img = img > self.min_gray

        # Remove small white regions
        open_img = ndimage.binary_opening(binary_img)
        # Remove small black hole
        close_img = ndimage.binary_closing(open_img)

        return close_img

    def transform(self, X):
        """ Transform image X using transform_hs() """
        return self.transform_hs(X)

# ----------------------- Tests --------------------------------#


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


def test_transform_hs_file():
    """Create image and transform it."""
    fpath = 'tests/4915_heatherglen_dr__houston__tx.jpg'
    img = mpimg.imread(fpath)
    fig, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img)
    img2 = transform_hs(img)
    axarr[1].imshow(img2, cmap='gray')
    plt.show()


def zest_transform_hs_rgb(fn_1=True):
    """Create image and transform it."""
    if fn_1:
        fpath = (
            "tests/"
            "39012_cl_consuelo_berges_19_3_b_cueto_santander_cantabria.jpg"
            )
    else:
        fpath = (
            'tests/39012_bo_corbanera_56_baj_monte_santander_cantabria.jpg'
            )
    img = mpimg.imread(fpath)
    fig, axarr = plt.subplots(1, 3)
    axarr[0].imshow(img)
    img2 = transform_hs(img, satur_min=0.4)
    axarr[1].imshow(img2, cmap='gray')
    img3 = transform_hs(img, gray_min=0)
    axarr[2].imshow(img3, cmap='gray')
    plt.show()


def test_transform_hs_file2():
    """Create image and transform it."""
    fpath = \
        'tests/39012_cl_consuelo_berges_19_3_b_cueto_santander_cantabria.jpg'
    img = mpimg.imread(fpath)
    fig, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img)
    img2 = transform_hs(img, satur_min=0.4)
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


def zest_HueSaturationTransformer(c=0.35, min_gray=None):
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
    hst = HueSaturationTransformer(min_gray=min_gray)
    hst.fit(img_train, Y)
    hst.set(hst.get())
    img_pred = hst.transform(img_test)

    print("save_attr: {} {}".format(len(hst.attributes), len(hst.get())))

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
    zest_HueSaturationTransformer(c=0.55, min_gray=0)


def test_HueSaturationTransformer2():
    zest_HueSaturationTransformer(c=0.35, min_gray=0.1)

if __name__ == '__main__':
    # test_transform_hs()
    # test_transform_hs_file()
    # test_transform_hs_file2()
    zest_transform_hs_rgb(fn_1=True)
    # zest_transform_hs_rgb(fn_1=False)
    # test_HueSaturationTransformer()
    # test_HueSaturationTransformer2()
