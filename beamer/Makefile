LATEXMK := latexmk
LATEXMK_OPTIONS = -lualatex -interaction=nonstopmode -file-line-error -output-directory=.output
LATEXINDENT := latexindent
LATEXINDENT_OPTIONS = -w -c=.output
CLEAN := rm
CLEAN_OPTIONS = -rf

ifneq (, $(shell which $(LATEXMK)))
	# if no latex environment is detected
	LATEXMK := $(shell which $(LATEXMK))
	LATEXINDENT := $(shell which $(LATEXINDENT))
	CLEAN := rm
else
	CONTAINER_NAME := texlive-container
	LATEXMK := docker exec $(CONTAINER_NAME) latexmk
	LATEXINDENT := docker exec $(CONTAINER_NAME) latexindent
	CLEAN := docker exec $(CONTAINER_NAME) rm

	# Check if container is running, start if not
	ifeq (, $(shell docker ps -q -f name=$(CONTAINER_NAME)))
		START_CONTAINER := docker run --rm -d -v $(PWD):/workdir -w /workdir --name $(CONTAINER_NAME) texlive/texlive tail -f /dev/null
	endif
endif

# match for root .tex files
TEX_FILES := $(wildcard *.tex)

.PHONY: lint clean all

all: clean lint $(TEX_FILES)

%.tex: FORCE
	$(START_CONTAINER)
	$(LATEXMK) $(LATEXMK_OPTIONS) $@

FORCE:

lint:
	find . -name "*.tex" -exec ${LATEXINDENT} ${LATEXINDENT_OPTIONS} {} \;

clean:
	$(CLEAN) $(CLEAN_OPTIONS) .output

stop:
	docker stop $(CONTAINER_NAME)
