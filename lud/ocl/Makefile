include ../../../common/make.config

#INC = -I$(OPENCL_INC) -I../common
#LIB = -L$(OPENCL_LIB)

#C compiler
CC = g++
CC_FLAGS = -g -O3 -Wall

# Xilinx OpenCL
XILINX_OPENCL := /opt/Xilinx/SDAccel/2015.4
XILINX_OPENCL_INC := $(XILINX_OPENCL)/runtime/include/1_2
XILINX_OPENCL_LIB := $(XILINX_OPENCL)/runtime/lib/x86_64

all: lud_gpu.exe \
     lud_fpga.exe

lud_gpu.exe: lud_gpu.cpp
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -o $@ lud_gpu.cpp ../common/common.c $(INC) $(LIB) -lOpenCL

lud_fpga.exe: lud_fpga.cpp
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) lud_fpga.cpp ../common/common.c -I../common -o $@  -I$(XILINX_OPENCL_INC) -L$(XILINX_OPENCL_LIB) -lxilinxopencl

clean:
	rm -f *.exe
