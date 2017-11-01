#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "gpuScan.h"

//all of your work will go in this file
//but you can change the tests in scan.cu
//for debugging purposes

void exclusiveScan(int * d_output, int length);

/*
 * exclusiveScan
 * Performs the exclusive scan on the GPU
 *
 * @param - d_output array that contains the input
 *          to use for the exclusive scan and holds
 *          the output of the execlusive scan
 * @param - length of array; always a power of 2
*/ 
void exclusiveScan(int * d_output, int length)
{
   //You'll need to add code here to launch
   //kernels.
   //Some of the work you'll need to do will
   //be on the CPU side.

}

/* gpuScan
 * This function is a wrapper for the exclusive scan that is 
 * performed on the GPU. It uses cudaMalloc to create an input/output
 * array on the GPU and copies the CPU array
 * to the GPU array. It initializes the timing functions and
 * then calls the exclusiveScan function to do the scan.
 * You should not modify this function.
 *
 * @param - output contains both the input to the scan and
 *          the output of the scan when complete
 * @param - length is the size of the output array
 */
double gpuScan(int * output, int length)
{
    int * d_output;
    float cpuMsecTime = -1;
    cudaEvent_t start_cpu, stop_cpu;

    //create input/output array for GPU
    CHECK(cudaMalloc((void **)&d_output, sizeof(int) * length));
    CHECK(cudaMemcpy(d_output, output, length * sizeof(int), 
               cudaMemcpyHostToDevice));

    //start the timing
    CHECK(cudaEventCreate(&start_cpu));
    CHECK(cudaEventCreate(&stop_cpu));
    CHECK(cudaEventRecord(start_cpu));
    
    //do the scan and wait for all threads to complete
    exclusiveScan(d_output, length);
    cudaThreadSynchronize();

    //stop the timing
    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu));
    CHECK(cudaEventElapsedTime(&cpuMsecTime, start_cpu, stop_cpu));
   
    //copy the output of the GPU to the CPU array
    cudaMemcpy(output, d_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    //release the space for the GPU array
    CHECK(cudaFree(d_output));

    return cpuMsecTime;
}

