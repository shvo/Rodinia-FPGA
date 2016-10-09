include ../../common/make.config
CC = gcc
CC_FLAGS = -std=c99

ifdef VERIFY
override VERIFY = -DVERIFY
endif

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

ifdef TIMER
override TIMER = -DTIMER
endif

hybridsort_gpu.exe: hybridsort.c bucketsort.h mergesort.h bucketsort.c mergesort.c
	$(CC) $(CC_FLAGS) -lOpenCL -o $@ $(VERIFY) $(OUTPUT) $(TIMER) -I$(OPENCL_INC) -I$(OPENCL_DIR)/shared/inc/ $(HISTO_WG_SIZE_0) $(BUCKET_WG_SIZE_0) $(BUCKET_WG_SIZE_1) $(MERGE_WG_SIZE_0) $(MERGE_WG_SIZE_1) hybridsort.c bucketsort.c mergesort.c -L$(OPENCL_LIB) -lm


clean:
	rm -f  *.exe
