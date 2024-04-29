"""
The following documents the pyfar filters. Visit the
:doc:`gallery:gallery/interactive/pyfar_filter_types`
for an overview of the different filters and the
:doc:`gallery:gallery/interactive/pyfar_filtering`
for more information on using pyfar filter objects.
"""

from .band_filter import (
    butterworth,
    chebyshev1,
    chebyshev2,
    elliptic,
    bessel,
    crossover,
    notch
)

from .audiofilter import (
    bell,
    high_shelve,
    low_shelve,
    high_shelve_cascade,
    low_shelve_cascade
)

from .fractional_octaves import (
    fractional_octave_bands,
    reconstructing_fractional_octave_bands,
    fractional_octave_frequencies
)

from .gammatone import (
    GammatoneBands,
    erb_frequencies
)


__all__ = [
    'butterworth',
    'chebyshev1',
    'chebyshev2',
    'elliptic',
    'bessel',
    'crossover',
    'notch',
    'bell',
    'high_shelve',
    'low_shelve',
    'high_shelve_cascade',
    'low_shelve_cascade',
    'fractional_octave_bands',
    'reconstructing_fractional_octave_bands',
    'fractional_octave_frequencies',
    'GammatoneBands',
    'erb_frequencies'
]
