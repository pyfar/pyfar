#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy>=1.14.0',
    'scipy>=1.5.0',
    'pyfftw',
    'matplotlib',
    'python-sofa>=0.2.0',
    'urllib3'
]

setup_requirements = ['pytest-runner', ]

test_requirements = [
    'pytest',
    'bumpversion',
    'wheel',
    'watchdog',
    'flake8',
    'tox',
    'coverage',
    'Sphinx',
    'twine',
]

setup(
    author="The pyfar developers",
    author_email='',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Scientists',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Project for data formats in acoustics.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pyfar',
    name='pyfar',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/pyfar/pyfar',
    version='0.1.0',
    zip_safe=False,
)
