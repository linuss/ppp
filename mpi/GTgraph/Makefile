include Makefile.var

all: sprng ssca rand rmat spec 
test-run: sprng ssca rand rmat spec run

ssca:
	(cd $(SSCADIR); $(MAKE); cd ..)

rand:
	(cd $(RANDDIR); $(MAKE); cd ..)

rmat:
	(cd $(RMATDIR); $(MAKE); cd ..)

spec:
	(cd $(SPECIALDIR); $(MAKE); cd ..)

sprng:
	(cd $(SPRNGDIR); $(MAKE); cd ..)

run:
	SSCA2/GTgraph-ssca2
	mv sample.gr SSCA2.gr
	random/GTgraph-rand
	mv sample.gr random.gr
	R-MAT/GTgraph-rmat
	mv sample.gr RMAT.gr
	special/GTgraph-special -t 0 -s 2 -d 4 
	mv sample.gr star.gr

clean: clean-ssca clean-rand clean-rmat clean-special clean-sprng clean-gen

clean-ssca:
	(cd $(SSCADIR); $(MAKE) clean; cd ..)

clean-rand:
	(cd $(RANDDIR); $(MAKE) clean; cd ..)

clean-rmat:
	(cd $(RMATDIR); $(MAKE) clean; cd ..)

clean-special:
	(cd $(SPECIALDIR); $(MAKE) clean; cd ..)

clean-sprng:
	(cd $(SPRNGDIR); $(MAKE) clean; cd ..)

clean-gen:
	rm -rf *.dim log
