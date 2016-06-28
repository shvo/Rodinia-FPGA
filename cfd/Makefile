include ../../common/make.config
#C compiler
CC = g++

FLAGS = -O3

SRC_GPU = euler3d_gpu.cpp

SRC_FPGA = euler3d_fpga.cpp

all: cfd_gpu.exe \
     cfd_fpga.exe

cfd_gpu.exe: $(SRC_GPU)
	$(CC) $(KERNEL_DIM) $(SRC_GPU) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL $(FLAGS) 

cfd_fpga.exe: $(SRC_FPGA)
	$(CC) $(KERNEL_DIM) $(SRC_FPGA) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL $(FLAGS) 

clean: $(SRC)
	rm -f *.exe result*
