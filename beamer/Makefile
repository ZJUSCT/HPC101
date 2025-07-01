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
	LATEXMK := docker run --rm -v $(PWD):/workdir -w /workdir texlive/texlive latexmk
	LATEXINDENT := docker run --rm -v $(PWD):/workdir -w /workdir texlive/texlive latexindent
	CLEAN := docker run --rm -v $(PWD):/workdir -w /workdir texlive/texlive rm
endif

# match for root .tex files
TEX_FILES := $(wildcard *.tex)

.PHONY: lint clean all

all: clean lint $(TEX_FILES)

%.tex: FORCE
	$(LATEXMK) $(LATEXMK_OPTIONS) $@
	# cp .output/$(basename $@).pdf .

FORCE:

lint:
	find . -name "*.tex" -exec ${LATEXINDENT} ${LATEXINDENT_OPTIONS} {} \;

clean:
	$(CLEAN) $(CLEAN_OPTIONS) .output
