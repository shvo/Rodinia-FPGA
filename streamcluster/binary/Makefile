XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:3.0
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : streamcluster_kernels.xclbin \

.PHONY : all

streamcluster_kernels.xclbin: ./Kernels.cl
	$(XOCC) $(CLFLAGS) $< -o $@ ;\
