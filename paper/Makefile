.PHONY: docker

.DEFAULT_GOAL := docker

docker:
	docker run --rm \
	    --volume $(PWD):/data \
	    --user $(shell id -u):$(shell id -g) \
	    --env JOURNAL=joss \
	    openjournals/inara
