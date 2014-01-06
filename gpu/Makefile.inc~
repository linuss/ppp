
PROJ_BASE	:= $(HOME)/src/teaching/PPP/gpu/2013-2014

INCLUDES	:= -I"$(PROJ_BASE)/include"
CUDA_LIBS	:= -L"$(CUDA_INSTALL_PATH)/lib64/"


NVCCFLAGS	:= -arch compute_20 -code sm_20 --ptxas-options=-v
ifneq ($(debug), 1)
	CFLAGS		:= -std=c++0x -O3 -g0 -Wall
	NVCCFLAGS	+= -O3
else
	CFLAGS		:= -std=c++0x -O0 -g3 -Wall
	NVCCFLAGS	:= -O0
endif

LDFLAGS         := -lm -lX11 -lpthread
CUDA_LDFLAGS	:= $(LDFLAGS) -lrt -lcudart

CC		:= g++
NVCC	:= nvcc
LINKER	:= g++

