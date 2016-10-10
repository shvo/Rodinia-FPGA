XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:3.0
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : pathfinder_dynproc_kernel.xclbin \

.PHONY : all

pathfinder_dynproc_kernel.xclbin: ./kernels.cl
	$(XOCC) $(CLFLAGS) $< -o $@ ;\
