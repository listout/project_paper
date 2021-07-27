OPTS += --resource-path=resources
OPTS += --pdf-engine=pdflatex
#OPTS += -H header.tex
OPTS += -V fontsize=10pt
OPTS += -s

slides.pdf: slides.md
	pandoc -t beamer $(OPTS) $< -o $@

clean:
	rm -fr slides.pdf
