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
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from scipy import ndimage
import matplotlib.patches as patches


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
    def __init__(self):
        pass

    def fit(self, X, y=None, shape=None):
        self.y = y
        self.shape = shape

    def transform(self, X):
        if self.y is None:
            return transform_hs(X)


def create_test_image(x1=10, y1=20, d=10):
    """Create test image with blue rect inside."""
    shape = (32, 64, 3)
    img = np.full(shape, 0.5)
    img[:, :30, 0] = 0.10
    img[:, 30:, 0] = 0.90
    img[y1:y1+d, x1:x1+d, 0] = 0.55
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


def test_HueSaturationTransformer():
    """Create image and transform it."""
    p1 = (10, 10, 10)
    p_train = (8, 8, 14)
    p2 = (40, 10, 10)
    img_train = create_test_image(x1=p1[0], y1=p1[1], d=p1[2])
    img_test = create_test_image(x1=p2[0], y1=p2[1], d=p2[2])
    fig, axarr = plt.subplots(3, 1)
    ax = axarr[0]
    ax.imshow(img_train, cmap='gray')
    show_patch(ax, p_train)
    axarr[1].imshow(img_test, cmap='gray')
    img = transform_hs(img_test)
    axarr[2].imshow(img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    # test_transform_hs()
    test_HueSaturationTransformer()
