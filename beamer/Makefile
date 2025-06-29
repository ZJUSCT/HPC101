LATEXMK := latexmk
LATEXMK_OPTIONS = -lualatex -interaction=nonstopmode -file-line-error -output-directory=.output
ifneq (, $(shell which $(LATEXMK)))
	LATEXMK := $(shell which $(LATEXMK))
else
	LATEXMK := docker run --rm -v $(PWD):/workdir -w /workdir texlive/texlive latexmk
endif
LATEXINDENT := latexindent
LATEXINDENT_OPTIONS = -w -c=.output
ifneq (, $(shell which $(LATEXINDENT)))
	LATEXINDENT := $(shell which $(LATEXINDENT))
else
	LATEXINDENT := docker run --rm -v $(PWD):/workdir -w /workdir texlive/texlive latexindent
endif

# match for root .tex files
TEX_FILES := $(wildcard *.tex)

.PHONY: lint clean all

all: clean lint $(TEX_FILES)

%.tex: FORCE
	$(LATEXMK) $(LATEXMK_OPTIONS) $@
	cp .output/$(basename $@).pdf .

FORCE:

lint:
	find . -name "*.tex" -exec ${LATEXINDENT} ${LATEXINDENT_OPTIONS} {} \;

clean:
	rm -rf .output
	rm -rf *.pdf
