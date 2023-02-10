#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy>=1.23.0',
    'scipy>=1.5.0',
    'matplotlib',
    'sofar>=0.1.2',
    'urllib3',
    'deepdiff',
    'soundfile>=0.11.0'
]

setup_requirements = ['pytest-runner', ]

test_requirements = [
    'pytest',
    'bump2version',
    'wheel',
    'watchdog',
    'flake8',
    'tox',
    'coverage',
    'Sphinx',
    'twine'
]

setup(
    author="The pyfar developers",
    author_email='info@pyfar.org',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],
    description="Project for data formats in acoustics.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='pyfar',
    name='pyfar',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url="https://pyfar.org/",
    download_url="https://pypi.org/project/pyfar/",
    project_urls={
        "Bug Tracker": "https://github.com/pyfar/pyfar/issues",
        "Documentation": "https://pyfar.readthedocs.io/",
        "Source Code": "https://github.com/pyfar/pyfar",
    },
    version='0.5.2',
    zip_safe=False,
    python_requires='>=3.8'
)
