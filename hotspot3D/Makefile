include ../../common/make.config

ifndef CC
	CC = gcc
endif

OUTPUT = *.out
CCFLAGS=-O3 -lm 
LIBS = -lOpenCL 

GPU_SRC = 3D_gpu.c CL_helper.c ./util/opencl/opencl.o
GPU_SRC_OPT1 = 3D_gpu_opt1.c CL_helper.c ./util/opencl/opencl.o
GPU_BURST_SRC = 3D_gpu_burst.c CL_helper.c ./util/opencl/opencl.o
FPGA_SRC = 3D_fpga.c CL_helper.c ./util/opencl/opencl.o
FPGA_DIRECT_SRC = 3D_fpga_direct.c CL_helper.c ./util/opencl/opencl.o
FPGA_BURST_SRC = 3D_fpga_burst.c CL_helper.c ./util/opencl/opencl.o
FPGA_SRC_OPT1 = 3D_fpga_opt1.c CL_helper.c ./util/opencl/opencl.o

all: hotspot3D_gpu.exe \
     hotspot3D_gpu_opt1.exe \
     hotspot3D_gpu_burst.exe \
     hotspot3D_fpga.exe \
     hotspot3D_fpga_direct.exe \
     hotspot3D_fpga_burst.exe \
     hotspot3D_fpga_opt1.exe

hotspot3D_gpu.exe: $(GPU_SRC)
	$(CC) $(GPU_SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -L$(OPENCL_LIB) -o $@

hotspot3D_gpu_opt1.exe: $(GPU_SRC_OPT1)
	$(CC) $(GPU_SRC_OPT1) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -L$(OPENCL_LIB) -o $@

hotspot3D_gpu_burst.exe: $(GPU_BURST_SRC)
	$(CC) $(GPU_BURST_SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -L$(OPENCL_LIB) -o $@

hotspot3D_fpga.exe: $(FPGA_SRC)
	$(CC) $(FPGA_SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -L$(OPENCL_LIB) -o $@

hotspot3D_fpga_direct.exe: $(FPGA_DIRECT_SRC)
	$(CC) $(FPGA_DIRECT_SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -L$(OPENCL_LIB) -o $@

hotspot3D_fpga_burst.exe: $(FPGA_BURST_SRC)
	$(CC) $(FPGA_BURST_SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -L$(OPENCL_LIB) -o $@

hotspot3D_fpga_opt1.exe: $(FPGA_SRC_OPT1)
	$(CC) $(FPGA_SRC_OPT1) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -L$(OPENCL_LIB) -o $@

opencl.o: ./util/opencl/opencl.c ./util/opencl/opencl.h
	gcc $< -o ./util/opencl/opencl.o -O3

clean:
	rm -f *.exe $(OUTPUT) 
