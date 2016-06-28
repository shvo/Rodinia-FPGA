XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:3.0
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : heartwall_default.xclbin \

.PHONY : all

heartwall_default.xclbin: ./kernel_gpu_opencl.cl
	$(XOCC) $(CLFLAGS) --nk kernel_gpu_opencl:1 -k kernel_gpu_opencl $< -o $@ ;\
