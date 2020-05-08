# haiopy

Haiopy Acoustics Input and Output in Python. An open and common base for Python programming in acoustics.

Getting Started
===============

There is no stable release and therefore no binary distribution yet. Hence, the
package needs to be installed from source.
When installing from source, we need to compile some C++ code using Cython. This will be handled automatically, but may introduce an additional step.

Requirements
============

- Python 3 (>= 3.6 recommended)
- Python packages: numpy, scipy, matplotlib

Installation
============

The sources for spharpy can be downloaded from the `git repository`_.

You can either clone the public repository

.. code-block:: console

    $ git clone git@github.com:mberz/haiopy.git

or download the code as zip archive.

You can install it with:

.. code-block:: console

    $ python setup.py install

The required packages to install the package can be installed using pip:

.. code-block:: console

    $ pip install -r requirements_dev.txt

When actively developing for the package it is recommended to install using the
develop option:

.. code-block:: console

    $ python setup.py develop

.. _git repository: https://github.com/mberz/haiopy

