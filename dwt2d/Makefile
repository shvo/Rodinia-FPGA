#all:
#	g++ -o components.o -I/usr/local/cuda-5.5/include -c components.cpp
#	g++ -o prog  -I/usr/local/cuda-5.5/include main.cpp components.o -lOpenCL
#
include ../../common/make.config


ifdef OUTPUT
	override OUTPUT = -DOUTPUT
endif


OUTPUT = -DOUTPUT

C_C = g++
#OCL_LIB = -lOpenCL
#OCL_INC = -I/usr/local/cuda-5.5/include 

all: dwt2d_gpu.exe \
  	 dwt2d_fpga.exe

components: 
	$(C_C) components.cpp -o components.o -I$(OPENCL_INC) -c

dwt2d_fpga.exe: 
	$(C_C) -o $@ main_fpga.cpp \
		-I$(OPENCL_INC) \
		$(components) \
		$(OUTPUT) \
		-L$(OPENCL_LIB)\
		-lOpenCL

dwt2d_gpu.exe: 
	$(C_C) -o $@ main_gpu.cpp \
		-I$(OPENCL_INC) \
		$(components) \
		$(OUTPUT) \
		-L$(OPENCL_LIB)\
		-lOpenCL

clean:	
	rm -rf *.o
	rm *.bmp.dwt.* *.exe

