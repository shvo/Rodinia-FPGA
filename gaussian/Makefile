include ../../common/make.config

#C compiler
CC = g++

GPU_SRC = clutils.cpp gaussianElim_gpu.cpp utils.cpp

FPGA_SRC = clutils.cpp gaussianElim_fpga.cpp utils.cpp

CC_FLAGS = -g -O3

all: gaussian_gpu.exe \
     gaussian_fpga.exe

gaussian_gpu.exe:
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -lOpenCL  -fopenmp $(GPU_SRC) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB)

gaussian_fpga.exe:
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -lOpenCL  -fopenmp $(FPGA_SRC) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB)
	
clean: 
	rm -f *.exe
