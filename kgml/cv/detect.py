#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev
# License:  BSD 3 clause
"""
    Object Detector
"""
from __future__ import division, print_function
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
import matplotlib.patches as patches
import cPickle as pickle
from util import rgb_to_hsv, hsv_to_rgb, img_norm
from features import rgb2gray
import cv2
from skimage.transform import pyramid_gaussian


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

        # check case if no one bin is selected
        if np.sum(sels) == 0:
            hist_ratio = hist_1 / hist_0
            hist_ratio[hist_0 < 0.001] = 0.0
            i_max = np.argmax(hist_ratio)
            sels[i_max] = 1

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


# ---------------- Object Detector -----------------------------#


def non_max_suppression(boxes, scores=None, overlapThresh=0.5):
    """ NonMaxSuppression (Malisiewicz et al.)
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is None:
        idxs = np.argsort(y2)
    else:
        assert len(scores) == len(boxes)
        idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                         np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    if scores is None:
        return boxes[pick].astype("int")
    else:
        return boxes[pick].astype("int"), scores[pick]


def pyramid(image, downscale=1.5, max_layer=100, minSize=(32, 32),
            gaussian=False,
            interpolation=cv2.INTER_AREA):
    """ Generate pyramid of the images.

    Links
    -----
    http://goo.gl/BLvoGb
    """
    if gaussian:
        for img in pyramid_gaussian(image, max_layer=100,
                                    downscale=downscale):
            if img.shape[0] < minSize[1] or img.shape[1] < minSize[0]:
                break
            yield img
    else:
        # yield the original image
        yield image

        # keep looping over the pyramid
        for _ in xrange(max_layer):
            # compute the new dimensions of the image and resize it
            w = int(image.shape[1] / downscale)
            h = int(image.shape[0] / downscale)
            image = cv2.resize(image, (w, h), interpolation=interpolation)

            # if the resized image does not meet the supplied minimum
            # size, then stop constructing the pyramid
            if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
                break

            # yield the next image in the pyramid
            yield image


def sliding_window(image, win_width=64, win_height=None, step_width=32,
                   step_height=None, adjust=True):
    """ Sliding window.

    Links
    -----
    goo.gl/RxIDE0
    """
    if win_height is None:
        win_height = win_width
    if step_height is None:
        step_height = step_width
    # slide a window across the image
    for y in xrange(0, image.shape[0], step_height):
        y_break = False
        for x in xrange(0, image.shape[1], step_width):
            x_break = False
            # yield the current window
            y_end = y + win_height
            x_end = x + win_width
            if adjust:
                if y_end == image.shape[0]:
                    y_break = True
                elif y_end > image.shape[0]:
                    y_end = image.shape[0]
                    y = y_end - win_height
                    y_break = True
                if x_end == image.shape[1]:
                    x_break = True
                elif x_end > image.shape[1]:
                    x_end = image.shape[1]
                    x = x_end - win_width
                    x_break = True
                if x < 0 or y < 0:
                    raise StopIteration('too small image')
            yield (x, y, image[y: y_end, x: x_end])
            if x_break:
                break
        if y_break:
            break
    raise StopIteration('end of image')


def sliding_window_multiscale(image, win_width=64, win_height=None,
                              shift=0.25, downscale=1.5, max_layer=100):
    if win_height is None:
        win_height = win_width
    step_width = int(win_width * shift)
    step_height = int(win_height * shift)
    windows, boxes = [], []
    # loop over the image pyramid
    for resized in pyramid(image, downscale=downscale, max_layer=max_layer):
        scale = (image.shape[0] / resized.shape[0] +
                 image.shape[1] / resized.shape[1]) / 2
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(
                resized, win_width=win_width, win_height=win_height,
                step_width=step_width, step_height=step_height):
            # if the window does not meet our desired window size, ignore it
            # print(x, y, window.shape)
            assert window.shape[0] == win_height and \
                window.shape[1] == win_width
            box = np.array([x, y, x + win_width, y + win_height], dtype=float)
            box *= scale
            windows.append(window)
            boxes.append(box.astype(int))
    return windows, boxes


def zest_sliding_window_multiscale2():
    # load the image
    fpath = 'data/4915_heatherglen_dr__houston__tx.jpg'
    image = cv2.imread(fpath)
    windows, boxes = sliding_window_multiscale(
        image, win_width=64, win_height=64,
        shift=0.25, downscale=1.2)
    print("boxes:", len(boxes))
    n = 3977
    box = boxes[n]
    x1, y1, x2, y2 = list(box.astype(int))
    clone = image.copy()
    cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
    fig, axarr = plt.subplots(1, 2)
    axarr[0].imshow(clone)
    axarr[1].imshow(windows[n])
    plt.show()
    clone = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = list(box.astype(int))
        cv2.rectangle(clone, (x1, y1), (x2, y2), (0, x2-x1, y2-y1), 2)
    plt.imshow(clone)
    plt.show()


def zest_sliding_window_multiscale():
    # load the image
    fpath = 'data/4915_heatherglen_dr__houston__tx.jpg'
    image = cv2.imread(fpath)
    (winW, winH) = (128, 64)
    # loop over the image pyramid
    for resized in pyramid(image, downscale=1.5):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(
                resized, win_width=winW, win_height=winH,
                step_width=100, step_height=70):
            # if the window does not meet our desired window size, ignore it
            print(x, y, window.shape[0], window.shape[1])
            assert window.shape[0] == winH and window.shape[1] == winW
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(200)


def zest_pyramid():
    # load the image
    fpath = 'data/4915_heatherglen_dr__houston__tx.jpg'
    image = cv2.imread(fpath)

    # METHOD #1: No smooth, just scaling.
    # loop over the image pyramid
    for (i, resized) in enumerate(pyramid(image, downscale=1.1)):
        # show the resized image
        cv2.imshow("Layer {}".format(i + 1), resized)
        cv2.waitKey(0)

    # close all windows
    cv2.destroyAllWindows()


def zest_pyramid2():
    # load the image
    fpath = 'data/4915_heatherglen_dr__houston__tx.jpg'
    image = cv2.imread(fpath)

    # METHOD #2: Smooth
    # loop over the image pyramid
    for (i, resized) in enumerate(pyramid(image, downscale=1.5,
                                          gaussian=True)):
        # show the resized image
        cv2.imshow("Layer {}".format(i + 1), resized)
        cv2.waitKey(0)

    # close all windows
    cv2.destroyAllWindows()


class ObjectDetector(BaseEstimator, ClassifierMixin):
    """ A detector of objects in the images.
    """
    def __init__(self, clf=None, threshold=0.5,
                 win_width=64, win_height=None,
                 shift=0.25, downscale=1.5, max_layer=0,
                 nms_threshold=None
                 ):
        self.clf = clf
        self.threshold = threshold
        self.win_width = win_width
        self.win_height = win_height
        self.shift = shift
        self.downscale = downscale
        self.max_layer = max_layer
        self.nms_threshold = nms_threshold

    def split(self, image):
        clone = np.asarray(image)
        windows, boxes = \
            sliding_window_multiscale(
                    clone, win_width=self.win_width,
                    win_height=self.win_height,
                    shift=self.shift, downscale=self.downscale,
                    max_layer=self.max_layer)
        return windows, boxes

    def detect(self, image):
        windows, boxes = self.split(image)
        y_pred_proba = self.clf.predict_proba(windows)[:, 1]
        i_found = np.where(y_pred_proba > self.threshold)[0]
        boxes = np.asarray(boxes)[i_found]
        scores = y_pred_proba[i_found]
        if len(boxes) > 0 and self.nms_threshold is not None:
            res = non_max_suppression(
                boxes=boxes, scores=scores,
                overlapThresh=self.nms_threshold)
            boxes, scores = res
        return boxes, scores

    def predict(self, images):
        y_pred = []
        for img in images:
            boxes, scores = self.detect(img)
            y_pred.append(len(boxes) > 0)
        return np.array(y_pred, dtype=int)

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
    if __name__ == '__main__':
        plt.show()


def test_transform_hs_file():
    """Create image and transform it."""
    fpath = 'data/4915_heatherglen_dr__houston__tx.jpg'
    img = mpimg.imread(fpath)
    fig, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img)
    img2 = transform_hs(img)
    axarr[1].imshow(img2, cmap='gray')
    if __name__ == '__main__':
        plt.show()


def zest_transform_hs_rgb(fn_1=True):
    """Create image and transform it."""
    if fn_1:
        fpath = (
            "data/"
            "39012_cl_consuelo_berges_19_3_b_cueto_santander_cantabria.jpg"
            )
    else:
        fpath = (
            'data/39012_bo_corbanera_56_baj_monte_santander_cantabria.jpg'
            )
    img = mpimg.imread(fpath)
    fig, axarr = plt.subplots(1, 3)
    axarr[0].imshow(img)
    img2 = transform_hs(img, satur_min=0.4)
    axarr[1].imshow(img2, cmap='gray')
    img3 = transform_hs(img, gray_min=0)
    axarr[2].imshow(img3, cmap='gray')
    if __name__ == '__main__':
        plt.show()


def test_transform_hs_file2():
    """Create image and transform it."""
    fpath = \
        'data/39012_cl_consuelo_berges_19_3_b_cueto_santander_cantabria.jpg'
    img = mpimg.imread(fpath)
    fig, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img)
    img2 = transform_hs(img, satur_min=0.4)
    axarr[1].imshow(img2, cmap='gray')
    if __name__ == '__main__':
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
    if __name__ == '__main__':
        plt.show()


def test_HueSaturationTransformer():
    zest_HueSaturationTransformer(c=0.55, min_gray=0)


def test_HueSaturationTransformer2():
    zest_HueSaturationTransformer(c=0.35, min_gray=0.1)

if __name__ == '__main__':
    # test_transform_hs()
    # test_transform_hs_file()
    # test_transform_hs_file2()
    # zest_transform_hs_rgb(fn_1=True)
    # zest_transform_hs_rgb(fn_1=False)
    # test_HueSaturationTransformer()
    # test_HueSaturationTransformer2()
    # zest_pyramid()
    # zest_pyramid2()
    # zest_sliding_window_multiscale()
    zest_sliding_window_multiscale2()
