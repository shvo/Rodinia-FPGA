#include ../../common/make.config

# C compiler
CC = g++
CC_FLAGS = -g -O2 

# OpenCL
XILINX_OPENCL := /opt/Xilinx/SDAccel/2015.4
OPENCL_INC := $(XILINX_OPENCL)/runtime/include/1_2
OPENCL_LIB := $(XILINX_OPENCL)/runtime/lib/x86_64

backprop_gpu.exe: backprop.o imagenet.o facetrain.o
	$(CC) $(CC_FLAGS) backprop_ocl_gpu.cpp backprop.o imagenet.o facetrain.o -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB) -lxilinxopencl

backprop_fpga.exe: backprop.o imagenet.o facetrain.o
	$(CC) $(CC_FLAGS) backprop_ocl_fpga.cpp backprop.o imagenet.o facetrain.o -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB) -lxilinxopencl

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c
	
backprop.o: backprop.c 
	$(CC) $(CC_FLAGS) backprop.c -c
	
facetrain.o: facetrain.c
	$(CC) $(CC_FLAGS) facetrain.c -c

imagenet.o: imagenet.c
	$(CC) $(CC_FLAGS) imagenet.c -c

clean:
	rm -f *.o *~ backprop*.exe *.linkinfo
