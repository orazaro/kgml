#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev and others
# License:  BSD 3 clause
"""
    Timers
"""
from __future__ import division, print_function
import time


class Timer(object):
    """ Timer class.
    Links
    -----
    http://goo.gl/amrLUJ
    """
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name, end=' ')
        print('Elapsed: %s' % (time.time() - self.tstart))

def test_Timer():
    with Timer('sleep 0.001'):
        time.sleep(0.001)
    with Timer('sleep 0.01'):
        time.sleep(0.01)


if __name__ == "__main__":
    test_Timer()
