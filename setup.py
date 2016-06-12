#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev
# License:  BSD 3 clause
from setuptools import setup, find_packages
import kgml

setup(
    name="kgml",
    version=kgml.__version__,
    packages=find_packages(),
    # scripts=['say_hello.py'],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    # install_requires=['docutils>=0.3'],

    package_data={
        # If any package contains *.txt files, include them:
        '': ['*.txt'],
        # And include any *.ipynb files found in the 'kgml' package, too:
        'kgml': ['*.ipynb'],
    },

    # metadata for upload to PyPI
    author="Oleg Razguliaev",
    author_email="oleg@razgulyaev.com",
    description="Machine learning addons to sklearn etc.",
    license="BSD 3 clause",
    keywords=["machine learning", "time series", "sklearn"],
    url="https://github.com/orazaro/kgml",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
)
