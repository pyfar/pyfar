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
