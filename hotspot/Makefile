include ../../common/make.config

CC = gcc
CC_FLAGS = -g -O3 -Wall

GPU_SRC = hotspot_gpu.c hotspot_gpu.h OpenCL_helper_library.c OpenCL_helper_library.h

FPGA_SRC = hotspot_fpga.c hotspot_fpga.h OpenCL_helper_library.c OpenCL_helper_library.h

all: hotspot_gpu.exe \
     hotspot_fpga.exe

hotspot_gpu.exe: $(GPU_SRC) 
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $(GPU_SRC) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL

hotspot_fpga.exe: $(FPGA_SRC) 
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $(FPGA_SRC) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL

clean:
	rm -f *.exe 
