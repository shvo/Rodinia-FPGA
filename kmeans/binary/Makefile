XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:3.0
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : kmeans_kernel_c_default.xclbin \
      kmeans_swap_default.xclbin \
      kemans_kernels.xclbin

.PHONY : all

kmeans_kernels.xclbin: ./kmeans.cl
	$(XOCC) $(CLFLAGS) $< -o $@ ;\

kmeans_kernel_c_default.xclbin: ./kmeans.cl
	$(XOCC) $(CLFLAGS) --nk kmeans_kernel_c:1 -k kmeans_kernel_c $< -o $@ ;\

kmeans_swap_default.xclbin: ./kmeans.cl
	$(XOCC) $(CLFLAGS) --nk kmeans_swap:1 -k kmeans_swap $< -o $@ ;\
