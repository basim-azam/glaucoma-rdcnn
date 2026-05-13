# Compile CD-Former paper.
# Defaults to pdflatex; switch to xelatex by setting LATEX=xelatex on the command line.

LATEX  ?= pdflatex
BIBTEX ?= bibtex
MAIN    = main

.PHONY: all clean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex tables.tex refs.bib
	$(LATEX) -interaction=nonstopmode $(MAIN).tex || true
	$(BIBTEX) $(MAIN) || true
	$(LATEX) -interaction=nonstopmode $(MAIN).tex || true
	$(LATEX) -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc
	rm -f $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).synctex.gz

view: $(MAIN).pdf
	@which xdg-open >/dev/null && xdg-open $(MAIN).pdf || open $(MAIN).pdf
