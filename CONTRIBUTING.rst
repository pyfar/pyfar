.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/pyfar/pyfar/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

pyfar could always use more documentation, whether as part of the
official pyfar docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/pyfar/pyfar/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `pyfar` for local development.

1. Fork the `pyfar` repo on GitHub.
2. Clone your fork locally::

    $ git clone https://github.com/pyfar/pyfar.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv pyfar
    $ cd pyfar/
    $ python setup.py develop

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the
   tests, including testing other Python versions with tox::

    $ flake8 pyfar tests
    $ python setup.py test or py.test
    $ tox

   To get flake8 and tox, just pip install them into your virtualenv.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 3.5 and 3.6. Check
   https://travis-ci.com/pyfar/pyfar/pull_requests
   and make sure that the tests pass for all supported Python versions.


Testing Guidelines
-----------------------
Test-Driven-Development (TDD) is the fundamental technique followed in developing the pyfar software packages to ensure a good software development practice. In principle, it is based on `three steps <https://martinfowler.com/bliki/TestDrivenDevelopment.html>`_

- Write a test for the next bit of functionality you want to add.
- Write the functional code until the test passes.
- Refactor both new and old code to make it well structured.

In the following, you'll find a more specific guideline. Note: these instructions are not generally applicable outside of pyfar.

- The main tool used for testing is `pytest <https://docs.pytest.org/en/stable/index.html>`_.
- All tests are located in the *tests/* folder.
- Avoid dependencies on other pyfar functionalities. This allows easier debugging in case of failing tests due to errorneous implementations. The recommended workflow is given in the following sections.

Fixtures
~~~~~~~~
"Software test fixtures initialize test functions. They provide a fixed baseline so that tests execute reliably and produce consistent, repeatable, results. Initialization may setup services, state, or other operating environments. These are accessed by test functions through arguments; for each fixture used by a test function there is typically a parameter (named after the fixture) in the test function’s definition." (from https://docs.pytest.org/en/stable/fixture.html)

- All fixtures are implemented in *conftest.py*, whick makes them automatically available to all tests. This prevents from implementing redundant, unreliable code in several test files.
- Define the variables used in the test only once, either in the test itself or, preferably, in the definition of the fixture. This assures consistency and prevents from failing tests due to the definition of variables  with the same purpose at different positions or in different files.

Stubs
~~~~~
In case of pyfar, mainly **state verification** is applied in the tests. This means that the outcome of a function is compared to a desired value (``assert ...``). For more information, it is reffered to `Martin Fowler's article <https://martinfowler.com/articles/mocksArentStubs.html.>`_.
To follow the principle of avoiding the dependency on other functionalities in case of objects, **stubs** are used. Stubs mimic the actual objects, but have minimum functionality and *fixed, well defined properties* used for assertions.

It requires a little more effort to implement stubs of the pyfar classes. Therefore, stub utilities are provided in *pyfar/testing/stub_utils.py* and imported in *confest.py*, where the actual stubs are implemented.

- Note: the stub utilities are not meant to be imported to test files directly or used for other purposes than testing. They solely provide functionality to create fixtures.
- The utilities simplify and harmonize testing within the pyfar package and improve the readability and reliability.
- The implementation as the private submodule ``pyfar.testing.stub_utils``  further allows the use of similar stubs in related packages with pyfar dependency (e.g. other packages from the pyfar family).
To get an idea of the recommended stub workflow have a look at the ``sine`` fixure in *conftest.py*.

**Pyfar Stubs as Dummies**

Beside the use of stubs replacing objects, it is highly recommended to use **stubs as dummies**. Dummies could provide some data or several related variables needed to call a certain function (i.e. time data and sampling rate), while the actual values are of no importance.

A good example is ´´test_signal_init´´ in *test_signal.py*.

**When Not to Use Stubs**

Sometimes, the dependency on another pyfar functionality is desired, so a stub makes no sense. Nevertheless, consider using a fixture, as for example done with the ``sine_signal`` fixture in *conftest.py*.

**Mocks**

Mocks are similar to stubs but used for **behavioral verification**. For example, a mock can replace a function or an object to check if it is called with correct parameters. A main motivation for using mocks is to avoid complex or time-consuming external dependencies, for example database queries.

- A typical use case of mocks in the pyfar context is hardware communication, for example reading and writing of large files or audio in- and output. These use cases are rare compared to tests performing state verification with stubs.
- In contrast to some other guidelines on mocks, external depencies do *not* need to be mocked in general. Failing tests due to changes in external packages are meaningful hints to modify the code.
- Examples of internal mocking can be found in *test_io.py*, indicated by the pytest ``@patch`` calls.

Pytest Tips
~~~~~~~~~~~
Pytest provides several, sophisticated functionalities which could reduce the effort of implementing tests.

- Similar tests executing the same code with different variables can be `parametrized <https://docs.pytest.org/en/stable/example/parametrize.html>`_. An example is ``test___eq___differInPoints`` in *test_coordinates.py*.

Feel free to add more recommendations on useful pytest functionalities here. Consider, that a trade-off between easy implemention and good readability of the tests needs to be found.


Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
Then run::

$ bumpversion patch # possible: major / minor / patch
$ git push
$ git push --tags

Travis will then deploy to PyPI if tests pass.