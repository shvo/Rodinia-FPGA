include ../../common/make.config

#C compiler
CC = g++

# Xilinx OpenCL
XILINX_OPENCL := /opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL_INC := $(XILINX_OPENCL)/runtime/include/1_2
XILINX_OPENCL_LIB := $(XILINX_OPENCL)/runtime/lib/x86_64

FLAGS = -O3
GPU_SRC = streamcluster_gpu.cpp
GPU_EXE = streamcluster_gpu.exe

FPGA_SRC = streamcluster_fpga.cpp
FPGA_EXE = streamcluster_fpga.exe

all : $(GPU_EXE) \
      $(FPGA_EXE) \


$(GPU_EXE):$(GPU_SRC)
	$(CC) -lOpenCL  $(GPU_SRC) -o $(GPU_EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) $(FLAGS)

$(FPGA_EXE):$(FPGA_SRC)
	$(CC) -lOpenCL  $(FPGA_SRC) -o $(FPGA_EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) $(FLAGS) -I$(XILINX_OPENCL_INC) -L$(XILINX_OPENCL_LIB) -lxilinxopencl

errmsg:$(SRC)
	$(CC) -lOpenCL $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -D ERRMSG $(FLAGS)
	
ptx:$(SRC)
	$(CC) -lOpenCL $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -D PTX_MSG $(FLAGS)
	
res:$(SRC)
	$(CC) -lOpenCL $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -D RES_MSG $(FLAGS)
	
clean: $(SRC)
	rm -f  *.exe  result*
