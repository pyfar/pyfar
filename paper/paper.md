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


# Additional information and future developments

- Example gallery with tutorials and use cases
- Open educational resources are available to support learning and teaching in acoustics and audio signal processing using `pyfar`.
- Additional packages in the `pyfar` ecosystem provide functionality for specific use cases, e.g., room acoustics analysis, spherical array signal processing, and reading and writing of _sofa_ files standardized in AES69.

# Acknowledgements

We gratefully acknowledge contributions by the open-source community in the form of bug-fixes, functionality additions, and usability or documentation improvements.

# References
