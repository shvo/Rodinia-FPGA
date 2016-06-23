XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:3.0
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : srad_extract_kernel_default.xclbin \
      srad_prepare_kernel_default.xclbin \
      srad_reduce_kernel_default.xclbin \
      srad_srad_kernel_default.xclbin \
      srad_srad2_kernel_default.xclbin \
      srad_compress_kernel_default.xclbin \

.PHONY : all

srad_extract_kernel_default.xclbin: ./kernel_gpu_opencl.cl
	$(XOCC) $(CLFLAGS) --nk extract_kernel:1 -k extract_kernel $< -o $@ ;\

srad_prepare_kernel_default.xclbin: ./kernel_gpu_opencl.cl
	$(XOCC) $(CLFLAGS) --nk prepare_kernel:1 -k prepare_kernel $< -o $@ ;\

srad_reduce_kernel_default.xclbin: ./kernel_gpu_opencl.cl
	$(XOCC) $(CLFLAGS) --nk reduce_kernel:1 -k reduce_kernel $< -o $@ ;\

srad_srad_kernel_default.xclbin: ./kernel_gpu_opencl.cl
	$(XOCC) $(CLFLAGS) --nk srad_kernel:1 -k srad_kernel $< -o $@ ;\

srad_srad2_kernel_default.xclbin: ./kernel_gpu_opencl.cl
	$(XOCC) $(CLFLAGS) --nk srad2_kernel:1 -k srad2_kernel $< -o $@ ;\

srad_compress_kernel_default.xclbin: ./kernel_gpu_opencl.cl
	$(XOCC) $(CLFLAGS) --nk compress_kernel:1 -k compress_kernel $< -o $@ ;\
