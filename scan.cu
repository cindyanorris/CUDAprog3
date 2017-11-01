#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "gpuScan.h"

//prototypes for functions in this file
void initOnes(int * array, int length);
void initRandom(int * array, int length);
void compare(int * h_output, int * d_output, int length);
float cpuScan(int * output, int length);

#define NUMTESTS 6
typedef struct
{
    int length;        //number of data elements
    float speedupGoal; //speedup that you should aim for
    const char * type; //how array is initialized
} testType;

testType tests[NUMTESTS] = {{1 << 18, 2.0, "ones"}, 
                            {1 << 18, 4.0, "random"}, //faster because gpu has "warmed up"
                            {1 << 21, 8.0, "ones"}, 
                            {1 << 21, 8.0, "random"},
                            {1 << 24, 16.0, "ones"},
                            {1 << 28, 19.0, "ones"}}; 
/*
   driver for the scan program.  
   The main calls the functions to perform the tests
   specified in the tests array.
*/
int main()
{
    int i;
    float cpuTime;
    float gpuTime;
    float speedup;
    printf("%10s\t%8s\t%8s\t%8s\t%8s\t%8s\n", "Length", "Data", "CPU ms", "GPU ms", "Speedup", "Goal");
    for (i = 0; i < NUMTESTS; i++)
    {
        int * input = (int *) malloc(sizeof(int) * tests[i].length);
        int * output = (int *) malloc(sizeof(int) * tests[i].length);
        int * d_output = (int *) malloc(sizeof(int) * tests[i].length);

        //initialize the array to all 1s 
        //or random small numbers
        if (strcmp(tests[i].type, "ones") == 0)
            initOnes(input, tests[i].length);
        else
            initOnes(input, tests[i].length);

        //for convenience, set the output to the input and then
        //the scan routines can just operate on the output
        memcpy(output, input, sizeof(int) * tests[i].length);
        memcpy(d_output, input, sizeof(int) * tests[i].length);

        //perform the scan using the CPU
        cpuTime = cpuScan(output, tests[i].length);       
        gpuTime = gpuScan(d_output, tests[i].length);       
        speedup = cpuTime / gpuTime;
   
        //make sure the gpuScan produced the correct results
        compare(output, d_output, tests[i].length);

        //print the output
        printf("%10d\t%8s\t%8.4f\t%8.4f\t%8.4f\t%8.1f\n", 
               tests[i].length, tests[i].type, cpuTime, gpuTime, 
               speedup, tests[i].speedupGoal);

        //free the dynamically allocated data
        free(input);
        free(output);
        free(d_output);
    }
}    

/*
   cpuScan
   Performs an exclusive scan on an array of integers
   with length elements.  An exclusive scan sets
   each element output[i] to the sum of elements
   input[0] ... input[i - 1]. output[0] is set to
   0; output[1] is set to input[0]; output[2] is
   set to input[0] + input[1].  
   For example, given the array A={1,4,6,8,2}, the 
   exclusive scan output={0,1,5,11,19}.
   Before this function is called, the output array
   is set to the input array thus the exclusive
   scan is performed in-place.

   The algorithm works by going through the array
   twice.  In an upsweep phase, pairs of elements
   are summed.  In the downsweep phase, more
   pairs are summed and swapping occurs to get
   element in the correct positions.
*/
float cpuScan(int * output, int length)
{
    cudaEvent_t start_cpu, stop_cpu;
    float cpuMsecTime = -1;

    //time the scan
    CHECK(cudaEventCreate(&start_cpu));
    CHECK(cudaEventCreate(&stop_cpu));
    CHECK(cudaEventRecord(start_cpu));

    // upsweep phase.
    for (int twod = 1; twod < length; twod *= 2)
    {
        int twod1 = twod * 2;
        /* parallelize this for */
        for (int i = 0; i < length; i += twod1)
        {
            output[i + twod1 - 1] += output[i + twod - 1];
        }
    }

    output[length - 1] = 0;

    // downsweep phase.
    for (int twod = length/2; twod >= 1; twod /= 2)
    {
        int twod1 = twod * 2;
        /* parallelize this for */
        for (int i = 0; i < length; i += twod1)
        {
            int t = output[i+twod-1];
            output[i + twod - 1] = output[i + twod1 - 1];
            output[i + twod1 - 1] += t; 
        }
    }

    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu));
    CHECK(cudaEventElapsedTime(&cpuMsecTime, start_cpu, stop_cpu));
    return cpuMsecTime;
}

/* 
   compare
   Compares two arrays of integers with length
   elements to see if the elements are equal.
   If the arrays differ, outputs an error
   message and exits the program.
*/
void compare(int * h_output, int * d_output, int length)
{
   for (int i = 0; i < length; i++)
   {
      if (h_output[i] != d_output[i])
      {
         printf("Compare failed: h_output[%d] = %d, d_output[%d] = %d\n", 
                i, h_output[i], i, d_output[i]);
         exit(1);
      }
   }
}

/* 
   initRandom
   initializes an array of integers of size
   length to ones
*/
void initOnes(int * array, int length)
{
   int i;
   for (i = 0; i < length; i++)
      array[i] = 1;
}

/* 
   initRandom
   initializes an array of integers of size
   length to random values between 0 and 4,
   inclusive
*/
void initRandom(int * array, int length)
{
   int i;
   for (i = 0; i < length; i++)
      array[i] = rand() % 5;
}

