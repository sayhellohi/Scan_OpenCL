#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include<time.h>

#include <CL/cl.h>

#define DATA_SIZE 1024
using namespace std;
ofstream outfile;

/*const char *ProgramSource = 
	"__kernel void add(__global float *input, int size) \n"\
"{\n"\
   " int thid = get_global_id(0); \n"\
   "float temp[2*size] = {0}; \n"\
   "int pout = 0, pin = 1; \n"\
   "temp[thid] = (thid > 0) ? input[thid-1] : 0; \n"\
   "barrier(CLK_GLOBAL_MEM_FENCE); \n"\
    "for(int offset = 1; offset < size; offset = offset *2){ \n"\
	"pout = 1-pout; \n"\
	"pin = 1-pout; \n"\
    "if(thid >= offset) { \n"\
    "   temp[pout*size + thid] = temp[pout*size thid] + temp[pin*size + thid - offset];\n"\
   " } else {\n"\
  "     temp[pout*size + thid] = temp[pin*size + thid];\n"\
 "   }\n"\
 "barrier(CLK_GLOBAL_MEM_FENCE); \n"\
"}\n"\
"output[thid] = temp[pout*n + thid]; \n"\
"} \n"\
" \n";
*/
const char *ProgramSource =
"__kernel void add(__global float *input, int size){\n"\
 "int thid = get_global_id(0); \n"\
 "for(int offset = 1;offset < size; offset = offset*2){ \n"\
 "barrier(CLK_GLOBAL_MEM_FENCE); \n"\
 "float shubh; \n"\
 "if(thid >= offset) { \n"\
 "shubh = input[thid - offset]; }\n"\
 "barrier(CLK_GLOBAL_MEM_FENCE); \n"\
 "if(thid >= offset) { \n"\
 "input[thid] = input [thid] + shubh; }\n"\
 "barrier(CLK_GLOBAL_MEM_FENCE); \n"\
 "} \n"\
 "}\n"\
	"\n";
	

 
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
cl_mem inputA;
outfile.open("shubham.txt");
size_t global;
size_t local;
 int i;
 float results[DATA_SIZE]={0};
clock_t start, ends;


float inputDataA[DATA_SIZE];
for(i=0;i<DATA_SIZE;i++)
	inputDataA[i] = (float)i;

if(clGetPlatformIDs(1, &platform_id, &num_of_platforms) != CL_SUCCESS)
{
	printf("Unable to get platform id\n");
	return 1;
}
 

if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &num_of_devices) != CL_SUCCESS)
{
printf("Unable to get device_id\n");
return 1;
}

properties[0]= CL_CONTEXT_PLATFORM;
properties[1]= (cl_context_properties) platform_id;
properties[2]= 0;
 
context = clCreateContext(properties,1,&device_id,NULL,NULL,&err);

command_queue = clCreateCommandQueue(context, device_id, 0, &err);

program = clCreateProgramWithSource(context,1,(const char **) &ProgramSource, NULL, &err);

if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS)
{
printf("Error building program\n");
return 1;
}

kernel = clCreateKernel(program, "add", &err);
 
inputA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * DATA_SIZE, NULL, NULL);

clEnqueueWriteBuffer(command_queue, inputA, CL_TRUE, 0, sizeof(float) * DATA_SIZE, inputDataA, 0, NULL, NULL);

start = clock(); 
int temp = DATA_SIZE;
clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputA);
clSetKernelArg(kernel, 1, sizeof(int), &temp);
global = DATA_SIZE;
local = DATA_SIZE;
clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
clFinish(command_queue);

clEnqueueReadBuffer(command_queue, inputA, CL_TRUE, 0, sizeof(float) *DATA_SIZE, results, 0, NULL, NULL);


printf("output: ");
 
for(i=0;i<DATA_SIZE; i++)
{
printf("%f ",results[i]);
outfile << results[i] << " ";
}

ends = clock();
double time_taken = ((double) (ends - start)) / CLK_TCK;
cout<<"Time taken is :"<<time_taken<<" Seconds"<<endl;
outfile<<"Time taken to run the code is: "<<(double)time_taken<<" Seconds"<<endl;
clReleaseMemObject(inputA);
clReleaseProgram(program);
clReleaseKernel(kernel);
clReleaseCommandQueue(command_queue);
clReleaseContext(context);
return 0;
}