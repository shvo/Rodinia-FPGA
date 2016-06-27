include ../../common/make.config

#C compiler
CC = g++

GPU_SRC = bfs_gpu.cpp timer.cc

FPGA_SRC = bfs_fpga.cpp timer.cc

CC_FLAGS = -g -O3

all: bfs_gpu.exe \
     bfs_fpga.exe

bfs_gpu.exe:$(SRC)
	$(CC) $(CC_FLAGS) -fopenmp $(GPU_SRC) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL 

bfs_fpga.exe:$(SRC)
	$(CC) $(CC_FLAGS) -fopenmp $(FPGA_SRC) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
