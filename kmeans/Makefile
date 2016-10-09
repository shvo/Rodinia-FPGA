include ../../common/make.config

# C compiler
CC = g++
CC_FLAGS = -g -O2 

# Xilinx OpenCL
XILINX_OPENCL := /opt/Xilinx/SDAccel/2015.4
XILINX_OPENCL_INC := $(XILINX_OPENCL)/runtime/include/1_2
XILINX_OPENCL_LIB := $(XILINX_OPENCL)/runtime/lib/x86_64

GPU_SRC = cluster.o getopt.o read_input.o kmeans_clustering.o rmse.o
FPGA_SRC = cluster.o getopt.o read_input.o kmeans_clustering.o rmse.o

all : kmeans_gpu.exe \
      kmeans_fpga.exe

kmeans_gpu.exe: $(GPU_SRC)
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -lOpenCL kmeans_gpu.cpp $(GPU_SRC) -o $@ -I$(OPENCL_INC) -I$(OPENCL_DIR)/shared/inc/ -L$(OPENCL_LIB)

kmeans_fpga.exe: $(FPGA_SRC)
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -lOpenCL kmeans_fpga.cpp $(FPGA_SRC) -o $@ -I$(XILINX_OPENCL_INC) -L$(XILINX_OPENCL_LIB) -lxilinxopencl

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

cluster.o: cluster.c 
	$(CC) $(CC_FLAGS) cluster.c -c

getopt.o: getopt.c 
	$(CC) $(CC_FLAGS) getopt.c -c

kmeans.o: kmeans.c 
	$(CC) $(CC_FLAGS) read_input.c -c

rmse.o: rmse.c
	$(CC) $(CC_FLAGS) rmse.c -c

clean:
	rm -f *.o *~ kmeans *.linkinfo *.exe
