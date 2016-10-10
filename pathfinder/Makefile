include ../../common/make.config
CXX = g++
DEBUGFLAGS = -g
# You will need to modify the following 2 lines
# by inserting the correct path to the OpenCL
# lib and inc directories.
CXXFLAGS = -I$(OPENCL_INC) -c $(DEBUGFLAGS)
LINKFLAGS = -L$(OPENCL_LIB)

# link objects(binaries) together
pathfinder_gpu.exe: main.o OpenCL_gpu.o
	$(CXX) main.o OpenCL_gpu.o $(LINKFLAGS) -o $@

# compile main function file into object (binary)
main.o: main.cpp OpenCL_gpu.cpp
	$(CXX) $(CXXFLAGS) main.cpp OpenCL_gpu.cpp

clean:
	clear
	@rm -f *.exe *.o *.out *~ *\#
