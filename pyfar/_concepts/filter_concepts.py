"""
Filter Types
------------

There are three types of Filter objects:

- **FIR Filter:** Finite Impulse Response (FIR) filters are non-recursive
  filters. FIR filters very flexible and can have arbitrary magnitude and
  phase responses.
- **IIR Filter:** Infinite Impulse Response (IIR) filters are recursive
  filters. The can achieve steeper filter slopes than FIR filters of the same
  order but are less flexible with respect to the phase response.
- **SOS Filter:** Second Order Section (SOS) filters are cascaded 2nd order
  recursive filters. They are often more robust against numerical errors than
  IIR filters of the same order.


Initializing a Filter Object
----------------------------

A filter object is initialized at least with the coefficients and a sampling
rate

.. code-block:: python

    import pyfar as pf
    filter = pf.FilterFIR([[2, -2]], 44100, state=None)


Applying a Filter Object
------------------------

To filter an audio signal, pass it to the filters process function

.. code-block:: python

    in = pf.Signal([1, 2, 3, 4], 44100)
    out = filter.process(in)

The output is ``out.time = [[2, 2, 2, 2]]`` and has the same number of samples
as the input.

The output will be the same no matter how often ``process`` is called. This
default behavior is often desired. In some cases, a different functionality can
be useful. For blockwise processing of input signals, the Filter object can
track the `state` of the filter. The initial state can be passed during
initialization or typical states can be set using

.. code-block:: python

    filter.init_state(in.cshape, state='zeros')

The above initializes the state with zeros, and if the filter is called with
blocks of the input

.. code-block:: python

    block_one = filter.process(in[0, 0:2])
    block_two = filter.process(in[0, 2:4])

the the blockwise output yields the same as the complete output ``out`` seen
above, i.e., ``[block_one, block_two] = [[2, 2, 2, 2]]``. This is the case
because initializing the state also makes the filter object track the state
across multiple calls of the ``process`` functions.

Another option for initialization is

.. code-block:: python

    filter.init_state(in.cshape, state='step')

which makes sure that the first sample of the output is the same as the first
sample of the input.

To disable tracking the state, call ``process`` with ``reset=True``

.. code-block:: python

    out = filter.process(in, reset=True)

or simply do not initialize the state at all as done by

.. code-block:: python

    filter = pf.FilterFIR([[2, -2]], 44100, state=None)


See :py:class:`~pyfar.classes.filter` for a complete documentation.
"""
