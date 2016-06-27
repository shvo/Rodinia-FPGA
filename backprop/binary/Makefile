XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:3.0
XOCC := $(XILINX_SDACCEL)/bin/xocc
CPP := g++

OPENCL_INC := $(XILINX_OPENCL)/runtime/include/1_2
OPENCL_LIB := $(XILINX_OPENCL)/runtime/lib/x86_64

CXXFLAGS := -Wall -Werror -std=c++11
CLFLAGS := -s --xdevice $(DSA)

PATH_TO_UTILS=./../../utilities
OPENCL_SDK=/usr/local/cuda
INCLUDE=-I${OPENCL_SDK}/include/CL -I${PATH_TO_UTILS}
LIB=-lOpenCL

all : backprop_kernel_default.xclbin \

.PHONY : all

backprop_kernel_default.xclbin: ./backprop_kernel.cl
	$(XOCC) $(CLFLAGS) --nk bpnn_layerforward_ocl:1 --nk bpnn_adjust_weights_ocl:1 $< -o $@ ;\
