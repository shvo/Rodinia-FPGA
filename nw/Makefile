include ../../common/make.config

#C compiler
CC = g++
CC_FLAGS = -g -O3 -Wall

# Xilinx OpenCL
XILINX_OPENCL := /opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL_INC := $(XILINX_OPENCL)/runtime/include/1_2
XILINX_OPENCL_LIB := $(XILINX_OPENCL)/runtime/lib/x86_64

all: nw_gpu.exe \
     nw_fpga.exe \

nw_gpu.exe: nw_gpu.c
	$(CC) ${KERNEL_DIM} $(CC_FLAGS) -o $@ nw_gpu.c -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL

nw_fpga.exe: nw_fpga.c
	$(CC) ${KERNEL_DIM} $(CC_FLAGS) nw_fpga.c -o $@ -I$(XILINX_OPENCL_INC) -L$(XILINX_OPENCL_LIB) -lxilinxopencl 

clean:
	rm -f *.exe
