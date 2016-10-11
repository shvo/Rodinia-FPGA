XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:3.0
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : find_ellipse_kernel.xclbin \
      track_ellipse_kernel.xclbin

.PHONY : all

find_ellipse_kernel.xclbin: ./find_ellipse_kernel.cl
	$(XOCC) $(CLFLAGS) $< -o $@ ;\

track_ellipse_kernel.xclbin: ./track_ellipse_kernel.cl
	$(XOCC) $(CLFLAGS) $< -o $@ ;\
