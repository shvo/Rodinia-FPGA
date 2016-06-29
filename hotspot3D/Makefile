include ../../common/make.config

ifndef CC
	CC = gcc
endif

OUTPUT = *.out
CCFLAGS=-O3 -lm 
LIBS = -lOpenCL 

GPU_SRC = 3D_gpu.c CL_helper.c ./util/opencl/opencl.o
FPGA_SRC = 3D_fpga.c CL_helper.c ./util/opencl/opencl.o

all: hotspot3D_gpu.exe \
     hotspot3D_fpga.exe \

hotspot3D_gpu.exe: $(GPU_SRC)
	$(CC) $(GPU_SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -L$(OPENCL_LIB) -o $@

hotspot3D_fpga.exe: $(FPGA_SRC)
	$(CC) $(FPGA_SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -L$(OPENCL_LIB) -o $@

opencl.o: ./util/opencl/opencl.c ./util/opencl/opencl.h
	gcc $< -o ./util/opencl/opencl.o -O3

clean:
	rm -f *.exe $(OUTPUT) 
