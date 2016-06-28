XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:3.0
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : cfd_c_CopySrcToComponents_default.xclbin \
      cfd_c_CopySrcToComponent_default.xclbin \
      cfd_cl_fdwt53Kernel_default.xclbin \

.PHONY : all

cfd_c_CopySrcToComponents_default.xclbin: ./com_dwt.cl
	$(XOCC) $(CLFLAGS) --nk c_CopySrcToComponents:1 -k c_CopySrcToComponents $< -o $@ ;\

cfd_c_CopySrcToComponent_default.xclbin: ./com_dwt.cl
	$(XOCC) $(CLFLAGS) --nk c_CopySrcToComponent:1 -k c_CopySrcToComponent $< -o $@ ;\

cfd_cl_fdwt53Kernel_default.xclbin: ./com_dwt.cl
	$(XOCC) $(CLFLAGS) --nk cl_fdwt53Kernel:1 -k cl_fdwt53Kernel $< -o $@ ;\
