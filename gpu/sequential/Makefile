
debug		:= 0
include		../Makefile.inc


all: clean compare darker histogram smooth

darker: darker.cpp darker.k.cpp
	$(CC) -c darker.k.cpp $(INCLUDES) $(CFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darkerSeq darker.cpp darker.k.o ../src/Timer.cpp $(INCLUDES) $(CFLAGS) $(LDFLAGS)
	rm -f ./darker.k.o

histogram: histogram.cpp histogram.k.cpp
	$(CC) -c histogram.k.cpp $(INCLUDES) $(CFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/histogramSeq histogram.cpp histogram.k.o ../src/Timer.cpp $(INCLUDES) $(CFLAGS) $(LDFLAGS)
	rm -f ./histogram.k.o

smooth: smooth.cpp smooth.k.cpp
	$(CC) -c smooth.k.cpp $(INCLUDES) $(CFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/smoothSeq smooth.cpp smooth.k.o ../src/Timer.cpp $(INCLUDES) $(CFLAGS) $(LDFLAGS)
	rm -f ./smooth.k.o	

compare: compare.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/compare compare.cpp $(INCLUDES) $(CFLAGS) $(LDFLAGS) 

clean:
	rm -f $(PROJ_BASE)/bin/darkerSeq
	rm -f $(PROJ_BASE)/bin/histogramSeq
	rm -f $(PROJ_BASE)/bin/smoothSeq
	rm -f $(PROJ_BASE)/bin/compare

