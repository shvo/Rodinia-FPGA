include ../../common/make.config

#C compiler
CC = g++

GPU_SRC = bfs_gpu.cpp timer.cc ./util/opencl/opencl.o

FPGA_SRC = bfs_fpga.cpp timer.cc ./util/opencl/opencl.o

CC_FLAGS = -g -O3

all: bfs_gpu.exe \
     bfs_fpga.exe

bfs_gpu.exe: $(GPU_SRC)
	$(CC) $(CC_FLAGS) -fopenmp $(GPU_SRC) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL 

bfs_fpga.exe: $(FPGA_SRC)
	$(CC) $(CC_FLAGS) -fopenmp $(FPGA_SRC) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL 

opencl.o: ./util/opencl/opencl.c ./util/opencl/opencl.h
	gcc $< -o ./util/opencl/opencl.o -O3

clean: $(SRC)
	rm -f *.exe result*
