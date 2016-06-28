XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:3.0
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : hotspot3D_default.xclbin \

.PHONY : all

hotspot3D_default.xclbin: ./hotspotKernel.cl
	$(XOCC) $(CLFLAGS) --nk hotspotOpt1:1 -k hotspotOpt1 $< -o $@ ;\
