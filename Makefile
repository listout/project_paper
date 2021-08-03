OPTS += --resource-path=resources
OPTS += --pdf-engine=pdflatex
#OPTS += --toc
OPTS += -V fontsize=10pt
OPTS += -s

slides.pdf: slides.md
	pandoc -t beamer $(OPTS) $< -o $@ --slide-level=2

clean:
	rm -fr slides.pdf
