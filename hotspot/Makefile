include ../../common/make.config

CC = gcc
CC_FLAGS = -g -O3 -Wall

GPU_SRC = hotspot.c hotspot.h OpenCL_helper_library.c OpenCL_helper_library.h

FPGA_SRC = hotspot.c hotspot.h OpenCL_helper_library.c OpenCL_helper_library.h

all: hotspot_gpu.exe \
     hotspot_fpga.exe

hotspot_gpu.exe: $(GPU_SRC) 
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -o $@ -I$(OPENCL_INC) OpenCL_helper_library.c hotspot.c -L$(OPENCL_LIB) -lOpenCL

hotspot_fpga.exe: $(FPGA_SRC) 
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -o $@ -I$(OPENCL_INC) OpenCL_helper_library.c hotspot.c -L$(OPENCL_LIB) -lOpenCL

clean:
	rm -f *.exe 
