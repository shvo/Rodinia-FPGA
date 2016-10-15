XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:3.0
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : particle_naive_kernel.xclbin \

.PHONY : all

particle_naive_kernel.xclbin: ./particle_naive.cl
	$(XOCC) $(CLFLAGS) $< -o $@ ;\
