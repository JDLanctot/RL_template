#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools
from distutils.core import setup

pkg_name = 'co2'
author = "Eshan Merali, Jordan Lanctot, Junyeong Kim"
author_email = "jordan.lanctot@torontomu.ca"

install_requires = ['abstractcp',
                    'gym',
                    'matplotlib',
                    'more-itertools',
                    'numpy',
                    'pandas',
                    'pyyaml',
                    'scipy',
                    'torch',
                    'tslearn',
                    'tqdm']

if __name__ == '__main__':
    setup(
        name=pkg_name.lower(),
        description="CO2 Machine Learning",
        author=author,
        author_email=author_email,
        packages=setuptools.find_packages(),
        python_requires='>=3.8',
        install_requires=install_requires
)
