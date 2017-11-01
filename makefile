CC = /usr/local/cuda-8.0//bin/nvcc
GENCODE_FLAGS = -arch=sm_30
CC_FLAGS = -c --compiler-options -Wall,-Wextra,-O3,-m64
NVCCFLAGS = -m64 

scan: scan.o gpuScan.o
	$(CC) $(GENCODE_FLAGS) scan.o gpuScan.o -o scan

scan.o: scan.cu CHECK.h gpuScan.h
	$(CC) $(CC_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) scan.cu -o scan.o

gpuScan.o: gpuScan.cu CHECK.h gpuScan.h
	$(CC) $(CC_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) gpuScan.cu -o gpuScan.o

clean:
	rm scan scan.o gpuScan.o
