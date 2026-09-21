---
title: 'Pyfar: Python package for acoustics research'
tags:
  - Python
  - signal processing
  - acoustics
authors:
  - name: Marco Berzborn
    orcid: 0000-0002-4421-1702
    equal-contrib: true
    affiliation: "1"
  - name: Fabian Brinkmann
    affiliation: "2"
    equal-contrib: true
  - name: Anne Heimes
    affiliation: "3"
    equal-contrib: true
  - name: Simon Kersten
    affiliation: "3"
    equal-contrib: true

affiliations:
  - name: Department of the Built Environment, Technical University of Eindhoven, The Netherlands
    index: 1
  - name: Audio Communication Group, Technische Universität Berlin, Germany
    index: 2
  - name: Institute for Hearing Technology and Acoustics, RWTH Aachen, Germany
    index: 3

date: 17 May 2024
bibliography: references.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

`Pyfar` is a Python package for researchers and students in acoustics and audio signal processing.
It provides well documented and tested implementations of common operations and algorithms.
At it's core, `pyfar` provides encapsulated audio, filter, and coordinate objects to facilitate convenient
handling of data and related meta-data.
Relevant operators are implemented for the respective classes to enable intuitive and readable code.
On top of that `pyfar` provides functionality for

- audio and measurement signal generation,
- signal and data processing,
- visualization,
- file I/O.

`Pyfar` is the base package of the pyfar ecosystem, which contains more specific packages that depend on pyfar to realise, for example, room acoustic analysis and spherical array processing.

# Statement of need

Researchers in acoustics and audio signal processing often need to implement similar algorithms and functionality.
Existing implementations are often developed at single institutions for in-house use and later shared as open source solutions, leading to duplicated work, slower research progress, and less reliable results.
Additionally, research software is often developed within a specific project or by researchers who are on temporary contracts, introducing the risk of discontinued maintenance and support.
A shared cross-institutional codebase can mitigate this risk and extend the lifetime and continuity of the software through a larger community.

`Pyfar` is a cross-institutional community effort to provide a common foundation for acoustics research in Python.
It provides a well-documented and tested package that researchers can use and contribute to.
`Pyfar` is designed to be user-friendly and intuitive, making it easier for researchers to implement their ideas and share their work with others.
Encapsulated data structures and relevant operators facilitate readable and easy-to-use code, which reduces the chance of errors.

# State of the field

Several tools exist that were all developed at a single institution or are no longer maintained

- [`ITA-Toolbox`](https://git.rwth-aachen.de/ita/toolbox) [@Berzborn_2017_ITAToolboxOpenSource] and [`AKTools`](https://github.com/f-brinkmann/AKtools) [@Brinkman_2017_AKtoolsOpenSoftware]
  - Lack comprehensive documentation
  - Quality assurance not guaranteed due to lacking unit tests
  - Require a proprietary MATLAB license
  - Developed at single institutions
  - Maintenance limited to bug fixes
- [`pytta`](https://github.com/PyTTaMaster/PyTTa) [@Fonseca_2019_PyTTaOpenSource]
  - Development has slowed down and maintenance is not guaranteed
  - Developed at single institution
- [`python-acoustics`](https://github.com/python-acoustics/python-acoustics)
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

4. **Reliability**: The `pyfar` base package and ecosystem are developed, maintained, and reviewed by researchers and students from the field of acoustics. This aims at ensuring a high quality in the initial implementation of functionality and algorithms, while continuous integration based on `pytest` [^5] and hosted on `circle ci` [^6] is installed to monitor and guarantee deployability across different Python versions.

[^1]: https://github.com/pyfar/spharpy
[^2]: https://github.com/pyfar/pyrato
[^3]: https://pyfar.readthedocs.org
[^4]: https://mybinder.org
[^5]: https://docs.pytest.org/en/latest/
[^6]: https://circleci.com


# Research impact statement

At the time of writing, the `pyfar` package has seen contributions by 21 developers (including the core team). As of spring 2026, the group of developers spans a total of 10 institutions.
Accordingly, it can be concluded that the core idea of developing `pyfar` as a cross-institutional community effort is successfully implemented.

Since its initial release in 2021 a total of 34 versions were released. The latest version is `0.8.1`, which was released on August 14, 2026.
In addition to an active developer community, users actively contribute to `pyfar` by reporting issues and suggesting features via the GitHub issue tracker.

`Pyfar` has been downloaded from PyPI over 254,000 times in total, out of which of which 91,000 downloads were in the past year, i.e. 2025. These numbers were obtained using the clickpy platform [^7]. Note that these numbers also include downloads by automated tools such as continuous integration pipelines, which may inflate the numbers. However, they still indicate a significant and active user base.

Due to a lack of an official citation for `pyfar` until this paper, it is difficult to estimate the total number of projects and research using `pyfar`.
We found 41 open source repositories and 13 software packages that rely on `pyfar` as a dependency [^8] including:

- [`bayesian_listner`](https://github.com/robaru/bayesian_listener_package): Bayesian auditory model for human localization performance.
- [`choras`](https://github.com/choras-org/CHORAS): Web-based platform for running and comparing room acoustics simulations.
- [`flamo`](https://github.com/gdalsanto/flamo): Open source library for frequency-domain differentiable audio processing.
- [`mesh2hrtf`](https://github.com/Any2HRTF/Mesh2HRTF) and [`mesh2scattering`](https://github.com/ahms5/Mesh2scattering): Boundary element-based simulation of head-related transfer functions and scattering coefficients.
- [`pyFDN`](https://github.com/artificial-audio/pyFDN): Building blocks for designing, simulating, and analysing Feedback Delay Networks (FDNs).
- [`QASTAnet`](https://github.com/Orange-OpenSource/QASTAnet): Metric for predicting global audio quality of 3D audio signals by Orange.
- [`sparrowpy`](https://github.com/sparrow-acoustics/sparrowpy): Sound propagation with acoustic radiosity for realistic outdoor worlds.
- [`spectacular`](https://github.com/acoular/spectacoular): GUI-based interface for microphone array signal processing.
- [`universal_transcoder`](https://github.com/DolbyLaboratories/universal_transcoder): A universal spatial audio transcoder by Dolby Laboratories.

Note that we omitted dependent packages from the the `pyfar` ecosystem in the list above.

Beyond the usage in research software detailed above, `pyfar` is also used in university teaching.  Since 2025, the `pyfar` community created and maintained a collection of open educational resources, which was introduced in @Brinkmann_2025_OpenEducationalResources. As of fall 2026, the collection includes coding assignments in the form of Jupyter notebooks for 3 different courses at M.Sc. level.

[^7]: https://clickpy.clickhouse.com/dashboard/pyfar
[^8]: Aggregated from https://github.com/pyfar/pyfar/network/dependents?dependent_type=PACKAGE and https://deps.dev/pypi/pyfar/0.8.1/dependents on Sept. 21st 2026.

# Additional information and future developments

The pyfar ecosystem, which is the overarching home and conceptual foundation of the `pyfar` base package is hosted on [pyfar.org](https://pyfar.org). The ecosystem bundles the documentation off all contained packages, offers and example gallery containing interactive Jupyter Notebooks to enable an intuitive extension to the package documentations through tutorials and example use cases, and educational resources for teaching acoustics and audio signal processing on a university level.

At the time of writing, the pyfar ecosystem also comprised the `pyrato` package for room acoustics analysis, `spharpy` for spherical array signal processing, and `sofar` for reading and writing of _sofa_ files standardized in AES69 [@AES69-2022].

# AI usage disclosure

GitHub Copilot was used as an auto-completion engine and occasionally to review GitHub pull requests. No generative AI tools were used in the writing of this manuscript, or the preparation of supporting materials.

# Acknowledgements

We gratefully acknowledge contributions by the open-source community in the form of bug-fixes, functionality additions, and usability or documentation improvements.

# References
