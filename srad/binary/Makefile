XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2015.4
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:2.1
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

all : srad_default.xclbin \

.PHONY : all

srad_default.xclbin: ./kernel_gpu_opencl.cl
	$(XOCC) $(CLFLAGS) --nk extract_kernel:1 \
	  		   --nk prepare_kernel:1 \
			   --nk reduce_kernel:1 \
			   --nk srad_kernel:1 \
			   --nk srad2_kernel:1 \
			   --nk compress_kernel:1 $< -o $@ ;\
