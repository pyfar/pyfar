.. raw:: html

    <h1 align="center">
    <img src="https://github.com/pyfar/gallery/raw/main/docs/resources/logos/pyfar_logos_fixed_size_pyfar.png" width="300">
    </h1><br>


.. image:: https://badge.fury.io/py/pyfar.svg
    :target: https://badge.fury.io/py/pyfar
.. image:: https://readthedocs.org/projects/pyfar/badge/?version=latest
    :target: https://pyfar.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://circleci.com/gh/pyfar/pyfar.svg?style=shield
    :target: https://circleci.com/gh/pyfar/pyfar
.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/pyfar/gallery/main?labpath=docs/gallery/interactive/pyfar_introduction.ipynb


The python package for acoustics research (pyfar) offers classes to store
audio data, filters, coordinates, and orientations. It also contains functions
for reading and writing audio data, as well as functions for processing,
generating, and plotting audio signals.

Getting Started
===============

The `pyfar workshop`_ gives an overview of the most important pyfar
functionality and is a good starting point. It is part of the
`pyfar example gallery`_ that also contains more specific and in-depth
examples that can be executed interactively without a local installation by
clicking the mybinder.org button on the respective example. The
`pyfar documentation`_ gives a detailed and complete overview of pyfar. All
these information are available from `pyfar.org`_.

Installation
============

Use pip to install pyfar

.. code-block:: console

    $ pip install pyfar

(Requires Python 3.8 or higher)

Audio file reading/writing is supported through `SoundFile`_, which is based on `libsndfile`_. On Windows and OS X, it will be installed automatically. On Linux, you need to install libsndfile using your distribution’s package manager, for example ``sudo apt-get install libsndfile1``.
If the installation fails, please check out the `help section`_.

Contributing
============

Check out the `contributing guidelines`_ if you want to become part of pyfar.

.. _pyfar workshop: https://mybinder.org/v2/gh/pyfar/gallery/main?labpath=docs/gallery/interactive/pyfar_introduction.ipynb
.. _pyfar example gallery: https://pyfar-gallery.readthedocs.io/en/latest/examples_gallery.html
.. _pyfar documentation: https://pyfar.readthedocs.io
.. _pyfar.org: https://pyfar.org
.. _SoundFile: https://python-soundfile.readthedocs.io
.. _libsndfile: http://www.mega-nerd.com/libsndfile/
.. _help section: https://pyfar-gallery.readthedocs.io/en/latest/help
.. _contributing guidelines: https://pyfar.readthedocs.io/en/stable/contributing.html
