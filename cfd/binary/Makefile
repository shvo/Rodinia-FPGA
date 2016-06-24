XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:3.0
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : cfd_memset_kernel_default.xclbin \
      cfd_initialize_variables_default.xclbin \
      cfd_compute_step_factor_default.xclbin \
      cfd_compute_flux_default.xclbin \
      cfd_time_step_default.xclbin \

.PHONY : all

cfd_memset_kernel_default.xclbin: ./Kernels.cl
	$(XOCC) $(CLFLAGS) --nk memset_kernel:1 -k memset_kernel $< -o $@ ;\

cfd_initialize_variables_default.xclbin: ./Kernels.cl
	$(XOCC) $(CLFLAGS) --nk initialize_variables:1 -k initialize_variables $< -o $@ ;\

cfd_compute_step_factor_default.xclbin: ./Kernels.cl
	$(XOCC) $(CLFLAGS) --nk compute_step_factor:1 -k compute_step_factor $< -o $@ ;\

cfd_compute_flux_default.xclbin: ./Kernels.cl
	$(XOCC) $(CLFLAGS) --nk compute_flux:1 -k compute_flux $< -o $@ ;\

cfd_time_step_default.xclbin: ./Kernels.cl
	$(XOCC) $(CLFLAGS) --nk time_step:1 -k time_step $< -o $@ ;\
