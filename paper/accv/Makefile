LATEX  ?= pdflatex
BIBTEX ?= bibtex
MAIN    = main

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex tables.tex refs.bib accv.sty accvabbrv.sty llncs.cls splncs04.bst
	$(LATEX) -interaction=nonstopmode $(MAIN).tex || true
	$(BIBTEX) $(MAIN) || true
	$(LATEX) -interaction=nonstopmode $(MAIN).tex || true
	$(LATEX) -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc
