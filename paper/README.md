# Journal of Open Source Software manuscript for _pyfar_

This folder contains the source code for the (Journal of Open Source Software) JOSS manuscript on _pyfar_.
The JOSS webpage contains guidelines on the [format of the paper](<https://joss.readthedocs.io/en/latest/paper.html>) as well as [submission requirements](https://joss.readthedocs.io/en/latest/submitting.html).

## Building the paper

The easiest way to build the paper locally is using Docker.
The makefile in this folder provides a target for that.
On UNIX systems building the paper can be done by running the following command in this folder:

```shell
make docker
```

Note that the _docker_ target is also the default target of the makefile, so simply running `make` also works:

Alternatively, the following command can be used directly:

```shell
docker run --rm \
    --volume $(PWD):/data \
    --user $(shell id -u):$(shell id -g) \
    --env JOURNAL=joss \
    openjournals/inara
```

## Template instructions

These are copied from the example paper template provided by JOSS.

### Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

### Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

### Figures

Figures can be included like this:

``![Caption for example figure.\label{fig:example}](figure.png)``

and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:

``![Caption for example figure.](figure.png){ width=20% }``

