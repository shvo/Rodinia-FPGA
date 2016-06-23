XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:3.0
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : BFS_1_default.xclbin \
      BFS_2_default.xclbin \

.PHONY : all

BFS_1_default.xclbin: ./Kernels.cl
	$(XOCC) $(CLFLAGS) --nk BFS_1:1 -k BFS_1 $< -o $@ ;\

BFS_2_default.xclbin: ./Kernels.cl
	$(XOCC) $(CLFLAGS) --nk BFS_2:1 -k BFS_2 $< -o $@ ;\
