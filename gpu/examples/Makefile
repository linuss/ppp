
debug		:= 0
include		../Makefile.inc


all: clean vectorAddCUDA vectorAddCL
 
vectorAddCUDA: vectorAdd.cu
	$(NVCC) -o ../bin/vectorAddCUDA vectorAdd.cu ../src/Timer.cpp $(INCLUDES) $(NVCCFLAGS)

vectorAddCL: vectorAdd.cl.cpp
	$(LINKER) -o ../bin/vectorAddCL vectorAdd.cl.cpp ../src/Timer.cpp $(INCLUDES) -I"$(OPENCL_INCLUDE)" -L"$(OPENCL_LIB)" -lOpenCL $(CFLAGS)

clean:
	rm -f ../bin/vectorAddCUDA
	rm -f ../bin/vectorAddCL

