include ../../common/make.config

# C compiler
CXX = g++
DEBUGFLAGS = -g

# Xilinx OpenCL
XILINX_OPENCL := /opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL_INC := $(XILINX_OPENCL)/runtime/include/1_2
XILINX_OPENCL_LIB := $(XILINX_OPENCL)/runtime/lib/x86_64

CXXFLAGS = -I$(OPENCL_INC) -c $(DEBUGFLAGS)
LINKFLAGS = -L$(OPENCL_LIB)

all: pathfinder_gpu.exe \
     pathfinder_fpga.exe \

# link objects(binaries) together
pathfinder_gpu.exe: main_gpu.o OpenCL.o
	$(CXX) main_gpu.o OpenCL.o $(LINKFLAGS) -o $@

pathfinder_fpga.exe: main_fpga.o OpenCL.o
	$(CXX) main_fpga.o OpenCL.o $(LINKFLAGS) -o $@ -I$(XILINX_OPENCL_INC) -L$(XILINX_OPENCL_LIB) -lxilinxopencl

# compile main function file into object (binary)
main_gpu.o: main_gpu.cpp OpenCL.cpp
	$(CXX) $(CXXFLAGS) main_gpu.cpp OpenCL.cpp

main_fpga.o: main_fpga.cpp OpenCL.cpp
	$(CXX) $(CXXFLAGS) main_fpga.cpp OpenCL.cpp
clean:
	clear
	@rm -f *.exe *.o *.out *~ *\#
