include ../../../common/make.config

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

# OpenCL SDK installation path      
#AMD_OPENCL_DIR = /home/mwb7w/AMD-APP-SDK-v2.5-TC1-lnx64/

# OpenCL include and library paths
#OPENCL_INC = $(AMD_OPENCL_DIR)/include
#OPENCL_LIB = $(AMD_OPENCL_DIR)/lib/x86_64

# Matrix library path
MATRIX_DIR = ../meschach_lib

# C compiler
CC = gcc
CC_FLAGS = -g -O3 -Wall -I$(MATRIX_DIR) -I$(OPENCL_INC)


all: leukocyte_gpu.exe \
     leukocyte_fpga.exe

leukocyte_gpu.exe: detect_main_gpu.o avilib.o find_ellipse.o find_ellipse_opencl_gpu.o track_ellipse.o track_ellipse_opencl_gpu.o misc_math.o OpenCL_helper_library.o $(MATRIX_DIR)/meschach.a
	$(CC) $(CC_FLAGS) -lm avilib.o find_ellipse.o find_ellipse_opencl_gpu.o track_ellipse.o track_ellipse_opencl_gpu.o misc_math.o OpenCL_helper_library.o detect_main_gpu.o -o $@ $(MATRIX_DIR)/meschach.a -L$(OPENCL_LIB) -lm -lOpenCL

leukocyte_fpga.exe: detect_main_fpga.o avilib.o find_ellipse.o find_ellipse_opencl_fpga.o track_ellipse.o track_ellipse_opencl_fpga.o misc_math.o OpenCL_helper_library.o $(MATRIX_DIR)/meschach.a
	$(CC) $(CC_FLAGS) -lm avilib.o find_ellipse.o find_ellipse_opencl_fpga.o track_ellipse.o track_ellipse_opencl_fpga.o misc_math.o OpenCL_helper_library.o detect_main_fpga.o -o $@ $(MATRIX_DIR)/meschach.a -L$(OPENCL_LIB) -lm -lOpenCL
%.o: %.[ch]
	$(CC) $(OUTPUT) $(CC_FLAGS) $< -c

detect_main_gpu.o: detect_main_gpu.c find_ellipse.h track_ellipse.h avilib.h

detect_main_fpga.o: detect_main_fpga.c find_ellipse.h track_ellipse.h avilib.h

find_ellipse_gpu.o: avilib.h

track_ellipse_gpu.o: find_ellipse.h avilib.h

find_ellipse_fpga.o: avilib.h

track_ellipse_fpga.o: find_ellipse.h avilib.h

$(MATRIX_DIR)/meschach.a:
	cd $(MATRIX_DIR); ./configure --with-all; make all; make clean

clean:
	rm -f *.o *~ *.exe
