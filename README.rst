=====
pyfar
=====

.. image:: https://badge.fury.io/py/pyfar.svg
    :target: https://badge.fury.io/py/pyfar
.. image:: https://readthedocs.org/projects/pyfar/badge/?version=latest
    :target: https://pyfar.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://circleci.com/gh/pyfar/pyfar.svg?style=shield
    :target: https://circleci.com/gh/pyfar/pyfar
.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/pyfar/gallery/main?filepath=docs/gallery


The python package for acoustics research (pyfar) offers classes to store
audio data, filters, coordinates, and orientations. It also contains common
functions for digital audio signal processing.

Getting Started
===============

Check out `pyfar.org`_ for a tour through the pyfar
universe, including complete documentation of this package and
the other packages.

Installation
============

Use pip to install pyfar

.. code-block:: console

    $ pip install pyfar

(Requires Python 3.8 or higher)

Audio file reading/writing is supported through `SoundFile`_, which is based on `libsndfile`_. On Windows and OS X, it will be installed automatically. On Linux, you need to install libsndfile using your distribution’s package manager, for example ``sudo apt-get install libsndfile1``.

Contributing
============

Refer to the `contribution guidelines`_ for more information.


.. _contribution guidelines: https://github.com/pyfar/pyfar/blob/develop/CONTRIBUTING.rst
.. _pyfar.org: https://pyfar.org
.. _read the docs: https://pyfar.readthedocs.io/en/latest
.. _SoundFile: https://pysoundfile.readthedocs.io/en/latest/
.. _libsndfile: http://www.mega-nerd.com/libsndfile/
