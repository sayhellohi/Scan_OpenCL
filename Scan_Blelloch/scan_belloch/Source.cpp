#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include<time.h>

#include <CL/cl.h>
 
#define DATA_SIZE 1023
using namespace std;
ofstream outfile;


const char *ProgramSource =
"__kernel void add(__global float *input, __global float *output, __global float *temp, int size){\n"\
 "int thid = get_global_id(0); \n"\
 "int offset = 1; \n"\
 "temp[2*thid] = input[2*thid]; \n"\
 "temp[2*thid+1] = input[2*thid+1]; \n"\
 "for(int d= size>>1; d>0; d >>= 1){ \n"\
 "barrier(CLK_GLOBAL_MEM_FENCE); \n"\
 "if(thid < d){ \n"\
 "int ai = offset*(2*thid + 1)-1; \n"\
 "int bi = offset*(2*thid + 2)-1; \n"\
 "temp[bi] += temp[ai]; } \n"\
 "offset = offset*2; \n"\
 "} \n"\
 "temp[size-1] = 0; \n"\
 "barrier(CLK_GLOBAL_MEM_FENCE); \n"\
 "for(int d = 1; d<size; d *= 2){ \n"\
 "offset >>= 1; barrier(CLK_GLOBAL_MEM_FENCE); \n"\
 "if(thid < d) { \n"\
 "int ai = offset*(2*thid+1)-1; int bi = offset*(2*thid+2)-1; \n"\
 "float t = temp[ai]; temp[ai] = temp[bi]; temp[bi] += t; }  \n"\
 "} \n"\
 "barrier(CLK_GLOBAL_MEM_FENCE); \n"\
 "output[2*thid] = temp[2*thid]; \n"\
 "output[2*thid+1] = temp[2*thid+1]; \n"\
 "}\n"\
	"\n";
	/*
 */


 
int main(void)
{
cl_context context;
cl_context_properties properties[3];
cl_kernel kernel;
cl_command_queue command_queue;
cl_program program;
cl_int err;
cl_uint num_of_platforms=0;
cl_platform_id platform_id;
cl_device_id device_id;
cl_uint num_of_devices=0;  
cl_mem inputA,inputB, output;
outfile.open("shubham.txt");
size_t global,loc;

float inputDataA[DATA_SIZE];
float results[2*DATA_SIZE]={0};

int i;
for(i=0; i<DATA_SIZE;i++)
{
	inputDataA[i] = (float)i;
}
clock_t start, ends;

if(clGetPlatformIDs(1, &platform_id, &num_of_platforms) != CL_SUCCESS)
{
	printf("Unable to get platform id\n");
	return 1;
}
 

// try to get a supported GPU device
if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &num_of_devices) != CL_SUCCESS)
{
//	printf("shbham");
printf("Unable to get device_id\n");
return 1;
}
 
// context properties list - must be terminated with 0
properties[0]= CL_CONTEXT_PLATFORM;
properties[1]= (cl_context_properties) platform_id;
properties[2]= 0;
 
// create a context with the GPU device
context = clCreateContext(properties,1,&device_id,NULL,NULL,&err);
 
// create command queue using the context and device
command_queue = clCreateCommandQueue(context, device_id, 0, &err);
 
// create a program from the kernel source code
program = clCreateProgramWithSource(context,1,(const char **) &ProgramSource, NULL, &err);
 
// compile the program
if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS)
{
printf("Error building program\n");
return 1;
}
 
// specify which kernel from the program to execute
kernel = clCreateKernel(program, "add", &err);
 
// create buffers for the input and ouput
 
inputA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * DATA_SIZE, NULL, NULL);
inputB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * DATA_SIZE*2, NULL, NULL);
output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * DATA_SIZE*2, NULL, NULL);
 
// load data into the input buffer
clEnqueueWriteBuffer(command_queue, inputA, CL_TRUE, 0, sizeof(float) * DATA_SIZE, inputDataA, 0, NULL, NULL);
clEnqueueWriteBuffer(command_queue, inputB, CL_TRUE, 0, sizeof(float) * DATA_SIZE*2, 0, 0, NULL, NULL);
clEnqueueWriteBuffer(command_queue, output, CL_TRUE, 0, sizeof(float) * DATA_SIZE*2, 0, 0, NULL, NULL);

int temp = DATA_SIZE;

start = clock();

// set the argument list for the kernel command
clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputA);
clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
clSetKernelArg(kernel, 2, sizeof(cl_mem), &inputB);
clSetKernelArg(kernel, 3, sizeof(int), &temp);

global=DATA_SIZE;
loc = DATA_SIZE;
// enqueue the kernel command for execution
clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global, &loc, 0, NULL, NULL);
clFinish(command_queue);
 
// copy the results from out of the output buffer
clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, sizeof(float) *DATA_SIZE, results, 0, NULL, NULL);
//clEnqueueReadBuffer(command_queue, inputB, CL_TRUE, 0, sizeof(float) *16, shubh, 0, NULL, NULL);

// print the results
printf("output: ");
 
for(i=0;i<DATA_SIZE; i++)
{
printf("%f ",results[i]);
outfile << results[i] << " ";
}
ends = clock();
double time_taken = ((double) (ends - start)) / CLK_TCK;
outfile << endl<<"Time taken is : "<< time_taken << endl;
/*for(i=0;i<16;i++)
{
outfile << shubh[i] <<" ";
}*/
// cleanup - release OpenCL resources
clReleaseMemObject(inputA);
clReleaseMemObject(inputB);
clReleaseMemObject(output);
clReleaseProgram(program);
clReleaseKernel(kernel);
clReleaseCommandQueue(command_queue);
clReleaseContext(context);
return 0;
}