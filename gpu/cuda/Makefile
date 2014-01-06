
debug		:= 0
include		../Makefile.inc


all: clean darker histogram smooth

darker: darker.cpp darker.cu
	$(NVCC) -c darker.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darkerCUDA darker.cpp darker.o ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./darker.o

histogram: histogram.cpp histogram.cu
	$(NVCC) -c histogram.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/histogramCUDA histogram.cpp histogram.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./histogram.o

smooth: smooth.cpp smooth.cu
	$(NVCC) -c smooth.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/smoothCUDA smooth.cpp smooth.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./smooth.o

clean:
	rm -f $(PROJ_BASE)/bin/darkerCUDA
	rm -f $(PROJ_BASE)/bin/histogramCUDA
	rm -f $(PROJ_BASE)/bin/smoothCUDA

