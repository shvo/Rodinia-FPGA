include ../../common/make.config

ifndef CC
	CC = gcc
endif

OUTPUT = *.out
CCFLAGS=-O3 -lm 
LIBS = -lOpenCL 

GPU_SRC = 3D.c CL_helper.c
FPGA_SRC = 3D.c CL_helper.c

all: hotspot3D_gpu.exe \
     hotspot3D_fpga.exe \

hotspot3D_gpu.exe: $(GPU_SRC)
	$(CC) $(GPU_SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -L$(OPENCL_LIB) -o $@

hotspot3D_fpga.exe: $(FPGA_SRC)
	$(CC) $(FPGA_SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -L$(OPENCL_LIB) -o $@

clean:
	rm -f *.exe $(OUTPUT) 
