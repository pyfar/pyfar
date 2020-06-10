======
haiopy
======

Haiopy Acoustics Input and Output in Python. An open and common base for Python programming in acoustics.

Getting Started
===============

There is no stable release and therefore no binary distribution yet. Hence, the
package needs to be installed from source.

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


Building the Documentation
==========================

Currently we do not yet host the documentation on a public server.
The documentation can however be built locally using Sphinx by
executing the make script inside the docs folder.

.. code-block:: console

    $ cd docs/
    $ make html

After Sphinx finishes the documentation you can open the generated html found in

.. code-block:: console

    $ docs/_build/index.html

using any browser.

.. _git repository: https://github.com/mberz/haiopy

