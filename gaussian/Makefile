include ../../common/make.config

#C compiler
CC = g++

GPU_SRC = clutils_gpu.cpp gaussianElim_gpu.cpp utils.cpp ./util/opencl/opencl.o

FPGA_SRC = clutils_fpga.cpp gaussianElim_fpga.cpp utils.cpp ./util/opencl/opencl.o

CC_FLAGS = -g -O3

all: gaussian_gpu.exe \
     gaussian_fpga.exe

gaussian_gpu.exe:
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -lOpenCL  -fopenmp $(GPU_SRC) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB)

gaussian_fpga.exe:
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -lOpenCL  -fopenmp $(FPGA_SRC) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB)
	
opencl.o: ./util/opencl/opencl.c ./util/opencl/opencl.h
	gcc $< -o ./util/opencl/opencl.o -O3

clean: 
	rm -f *.exe
