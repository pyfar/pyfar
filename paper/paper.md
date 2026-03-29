---
title: 'Pyfar: A Python package and ecosystem for acoustics research'
tags:
  - Python
  - signal processing
  - acoustics
authors:
  - name: Marco Berzborn
    orcid: 0000-0002-4421-1702
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Department of the Built Environment, Technical University of Eindhoven, The Netherlands
   index: 1
date: 17 May 2024
bibliography: references.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

`Pyfar` is a Python package and ecosystem for researchers in acoustics and audio signal processing.
It provides well documented and tested implementations of common operations and algorithms.
At it's core, `pyfar` provides encapsulated audio, filter, and coordinate objects to facilitate convenient
handling of data and related meta-data.
Relevant operators are implemented for the respective classes to enable intuitive and readable code.
On top of that `pyfar` provides functionality for

- audio and measurement signal generation,
- signal and data processing,
- visualization,
- file I/O.

# Statement of need

- Researchers in acoustics and audio signal processing often need to implement similar algorithms and functionality.
- Existing implementations are often developed at single institutions for in-house use and later shared as open source solutions.
- This leads to duplicated work, slower research progress, and less reliable results.
- `Pyfar` is a cross-institutional community effort to provide a common foundation for acoustics research in Python.
- `Pyfar` provides a well-documented and tested package that researchers can use and contribute to.
- `Pyfar` is designed to be user-friendly and intuitive, making it easier for researchers to implement their ideas and share their work with others.
- Encapsulated data structures and relevant operators facilitate readable and easy-to-use code which reduces the chance of errors.

# State of the field

Several tools exist (most of them are historically based on MATLAB)

1. `ITA-Toolbox` (https://git.rwth-aachen.de/ita/toolbox) for MATLAB [@Berzborn_2017_ITAToolboxOpenSource]
    - Very comprehensive, but not easy to navigate and use even though most functionality is separated into modules
    - lacks good documentation and only has minimal examples
    - Extensively used and therefore tested
    - Quality assurance not guaranteed due to lacking unit tests
    - Requires a MATLAB license, which can be a barrier for some researchers and institutions
    - Developed at single institution
2. `AKTools` (https://github.com/f-brinkmann/AKtools) [@Brinkman_2017_AKtoolsOpenSoftware]
    - Not as comprehensive as ITA-Toolbox, but easier to navigate
    - Lacks good online documentation
    - Small user-base
    - Developed at single institution
3. `pytta` (https://github.com/PyTTaMaster/PyTTa) [@Fonseca_2019_PyTTaOpenSource]
    - Development has slowed down and maintenance is not guaranteed
    - Developed at single institution
4. `python-acoustics` (https://github.com/python-acoustics/python-acoustics)
    - Comprehensive, modular structure
    - Archived and no longer maintained

Development of `Pyfar` was started with the primary idea to combine the efforts
of multiple working groups and institutions into a shared codebase and
ecosystem. The initiative was jointly started by some of the developers and
maintainers of the above mentioned `ITA-Toolbox` and `AKTools`.
`Pyfar` was completely redesigned from the code to allow a more modular
structure and to be more user-friendly and intuitive to use.
Integration into or extension of existing packages was not considered a
viable option as existing packages were built at single institutions and
therefore not designed to be sufficiently modular and flexible.

# Software design

The `pyfar` ecosystem and base package are designed with the following core concepts in mind:

1. **Encapsulation**: `Pyfar` provides encapsulated data structures for audio data, filters, and coordinates, as well as modifications such as rotations. This allows to store relevant meta-data (e.g. the sampling rate of an audio signal, the normalization of the Fourier spectrum, or user- defined comments) alongside the data itself. Further, relevant operators are implemented, allowing for intuitive modifications of the data such as summing two audio signals in the time domain by simply using the `+` operator. Additionally, most objects provide methods to convert between different representations of the data such as conversions between the time and frequency domain for audio signals, or conversion between Cartesian, cylindrical, and spherical coordinates for coordinate objects.
Functionality for digital signal processing and other data manipulations are primarily immplemented as functions that operate on the respective data objects.
This design allows for more intuitive and readable code but also reduces the chance of errors as relevant meta-data is stored and handled together with the data itself.
From a maintainers perspective, this design also has the benefit of well defined and consistent interfaces between functions and data structures.

2. **Modularity**: All packages and sub-packages are designed to be as modular and independent as possible. This avoids tight coupling between different parts of the codebase. This allows users to install and use packages relevant to their research and therefore makes the ecosystem more accessible. A significant benefit of this modular approach is improved maintainability and extensibility. The `pyfar` base package implements core data structures and functionality shared across the entire ecosystem. Very specialized functionality is implemented in separate packages, which rely on the base package as dependency. Examples are the packages `spharpy`[^1] and `pyrato`[^2], which implement functionality for spherical array signal processing and room acoustics analysis, respectively.

3. **Usability**: In addition to the user-friendly encapsulation of data structures and easy to navigate modular design, `pyfar` strives to provide extensive documentation of all functionality via the Sphinx documentation framework. The documentation is available online via the platform `readthedocs.org` [^3]. Examples are included as part of the API-documentation and additionally a growing number of application examples in the form of interactive Jupyter notebooks are provided. The Jupyter notebooks are organized in a gallery which further supports interactive execution via the online computation platform mybinder.org [^4].

[^1]: https://github.com/pyfar/spharpy
[^2]: https://github.com/pyfar/pyrato
[^3]: https://pyfar.readthedocs.org
[^4]: https://mybinder.org

# Additional information and future developments

- Example gallery with tutorials and use cases
- Open educational resources are available to support learning and teaching in acoustics and audio signal processing using `pyfar`.
- Additional packages in the `pyfar` ecosystem provide functionality for specific use cases, e.g., room acoustics analysis, spherical array signal processing, and reading and writing of _sofa_ files standardized in AES69.

# Acknowledgements

We gratefully acknowledge contributions by the open-source community in the form of bug-fixes, functionality additions, and usability or documentation improvements.

# References
