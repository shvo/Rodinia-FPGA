include ../../common/make.config
#OPENCL_DIR = /af21/den4gr/NVIDIA_GPU_Computing_SDK/

# C compiler
CC = g++
CC_FLAGS = -O2 
#INCLUDES = -I$(OPENCL_DIR)/OpenCL/common/inc -I$(OPENCL_DIR)/shared/inc/
INCLUDES = -I$(OPENCL_INC)

# Xilinx OpenCL
XILINX_OPENCL := /opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL_INC := $(XILINX_OPENCL)/runtime/include/1_2
XILINX_OPENCL_LIB := $(XILINX_OPENCL)/runtime/lib/x86_64

ALL: OCL_particlefilter_naive OCL_particlefilter_double OCL_particlefilter_single \
     OCL_particlefilter_naive_fpga.exe \

OCL_particlefilter_naive: ex_particle_OCL_naive_seq.o
	$(CC) $(CC_FLAGS) ex_particle_OCL_naive_seq.o $(INCLUDES) -L$(OPENCL_LIB) -lOpenCL -o OCL_particlefilter_naive.exe

OCL_particlefilter_double: ex_particle_OCL_double_seq.o
	$(CC) $(CC_FLAGS) ex_particle_OCL_double_seq.o $(INCLUDES)  -L$(OPENCL_LIB) -lOpenCL -o OCL_particlefilter_double.exe 

OCL_particlefilter_single: ex_particle_OCL_single_seq.o
	$(CC) $(CC_FLAGS) ex_particle_OCL_single_seq.o $(INCLUDES)  -L$(OPENCL_LIB) -lOpenCL -o OCL_particlefilter_single.exe 

OCL_particlefilter_naive_fpga.exe: ex_particle_OCL_naive_seq_fpga.o
	$(CC) $(CC_FLAGS) ex_particle_OCL_naive_seq_fpga.o -o $@ -I$(XILINX_OPENCL_INC) -L$(XILINX_OPENCL_LIB) -lxilinxopencl


%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

ex_particle_OCL_naive_seq.o: ex_particle_OCL_naive_seq.cpp
	$(CC) $(CC_FLAGS) $(INCLUDES) ex_particle_OCL_naive_seq.cpp -c

ex_particle_OCL_double_seq.o: ex_particle_OCL_double_seq.cpp
	$(CC) $(CC_FLAGS) $(INCLUDES) ex_particle_OCL_double_seq.cpp -c

ex_particle_OCL_single_seq.o: ex_particle_OCL_single_seq.cpp
	$(CC) $(CC_FLAGS) $(INCLUDES) ex_particle_OCL_single_seq.cpp -c

ex_particle_OCL_naive_seq_fpga.o: ex_particle_OCL_naive_seq_fpga.cpp
	$(CC) $(CC_FLAGS) $(INCLUDES) ex_particle_OCL_naive_seq_fpga.cpp -c

clean:
	rm -f *.o *~ *.exe *.linkinfo
