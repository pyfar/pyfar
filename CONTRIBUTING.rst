.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given. The following helps you to start
contributing specifically to pyfar. Please also consider the
`general contributing guidelines`_ for example regarding the style
of code and documentation and some helpful hints.

Types of Contributions
----------------------

Report Bugs or Suggest Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The best place for this is https://github.com/pyfar/pyfar/issues.

Fix Bugs or Implement Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Look through https://github.com/pyfar/pyfar/issues for bugs or feature request
and contact us or comment if you are interested in implementing.

Write Documentation
~~~~~~~~~~~~~~~~~~~

pyfar could always use more documentation, whether as part of the
official pyfar docs, in docstrings, or even on the web in blog posts,
articles, and such.

Get Started!
------------

Ready to contribute? Here's how to set up `pyfar` for local development using the command-line interface. Note that several alternative user interfaces exist, e.g., the Git GUI, `GitHub Desktop <https://desktop.github.com/>`_, extensions in `Visual Studio Code <https://code.visualstudio.com/>`_ ...

1. `Fork <https://docs.github.com/en/get-started/quickstart/fork-a-repo/>`_ the `pyfar` repo on GitHub.
2. Clone your fork locally and cd into the pyfar directory::

    $ git clone https://github.com/YOUR_USERNAME/pyfar.git
    $ cd pyfar

3. Install your local copy into a virtualenv. Assuming you have Anaconda or Miniconda installed, this is how you set up your fork for local development::

    $ conda create --name pyfar python
    $ conda activate pyfar
    $ conda install pip
    $ pip install -e .
    $ pip install -r requirements_dev.txt

4. Create a branch for local development. Indicate the intention of your branch in its respective name (i.e. `feature/branch-name` or `bugfix/branch-name`)::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the
   tests::

    $ flake8 pyfar tests
    $ pytest

   flake8 test must pass without any warnings for `./pyfar` and `./tests` using the default or a stricter configuration. Flake8 ignores `E123/E133, E226` and `E241/E242` by default. If necessary adjust your flake8 and linting configuration in your IDE accordingly.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request on the develop branch through the GitHub website.


.. _general contributing guidelines: https://pyfar-gallery.readthedocs.io/en/latest/contribute/index.html
