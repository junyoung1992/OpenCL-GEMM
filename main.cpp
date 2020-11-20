#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <time.h>
#include <math.h>

char* get_source_code(const char* file_name, size_t* len);
void vec_mul_seq(const float* A, const float* B, float* C, const int ROW_A, const int COL_A, const int COL_B, int type);
void vec_mul_opencl(const float* A, const float* B, float* C, const int ROW_A, const int COL_A, const int COL_B, int type);

clock_t start, end;
size_t kernel_source_size;
const char* kernel_source = get_source_code("kernel.cl", &kernel_source_size);
const int TS{ 16 }, WPT{ 8 };

int main() {
	float* A, * B, * C_seq, * C_opencl;
	int equal{ 1 };
	const int ROW_A{ 1024 }, COL_A{ 1024 }, ROW_B{ 1024 }, COL_B{ 1024 };
	
	A = (float*)malloc(sizeof(float) * ROW_A * COL_A);
	B = (float*)malloc(sizeof(float) * ROW_B * COL_B);
	C_seq = (float*)malloc(sizeof(float) * ROW_A * COL_B);
	C_opencl = (float*)malloc(sizeof(float) * ROW_A * COL_B);

	srand(time(NULL));
	for (int i = 0; i < ROW_A * COL_A; i++) A[i] = float(rand() % 100) / 100;
	for (int i = 0; i < ROW_B * COL_B; i++) B[i] = float(rand() % 100) / 100;

	printf("Sequential version...\n");
	vec_mul_seq(A, B, C_seq, ROW_A, COL_A, COL_B, 1);
	printf("\n");

	printf("OpenCL version...\n");
	for (int i = 1; i <= 4; i++)
		vec_mul_opencl(A, B, C_opencl, ROW_A, COL_A, COL_B, i);

	for (int i = 0; i < ROW_A * COL_B; i++)
		if (fabs(C_seq[i] - C_opencl[i]) > 0.001) {
			printf("%d\t%f\t%f\n", i, C_seq[i], C_opencl[i]);
			equal = 0;
			break;
		}

	if (equal == 1) printf("\nSequential version == OpenCL version\n");
	else printf("\nSequential version != OpenCL version\n");

	return 0;
}

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

char* get_source_code(const char* file_name, size_t* len) {
	char* source_code;
	char buf[2] = "\0";
	int cnt = 0;
	size_t length;
	FILE* file = fopen(file_name, "r");

	if (file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}

	fseek(file, 0, SEEK_END);
	length = (size_t)ftell(file);
	rewind(file);

	source_code = (char*)malloc(length + 1);
	fread(source_code, length, 1, file);

	for (int i = 0; i < length; i++) {
		buf[0] = source_code[i];
		if (buf[0] == '\n') {
			cnt++;
		}
	}

	fclose(file);

	return source_code;
}

void vec_mul_seq(const float* A, const float* B, float* C,
	const int ROW_A, const int COL_A, const int COL_B,
	int type) {

	start = clock();
	for (int j = 0; j < ROW_A; j++)
		for (int i = 0; i < COL_B; i++) {
			float sum = 0.0f;
			for (int k = 0; k < COL_A; k++)
				sum += A[j * COL_A + k] * B[k * COL_B + i];
			C[j * COL_B + i] = sum;
		}

	printf("vec_mul_seq_%d\tElapsed time: %f sec\n", type, (double)(clock() - start) / CLOCKS_PER_SEC);
}

void vec_mul_opencl(const float* A, const float* B, float* C,
	const int ROW_A, const int COL_A, const int COL_B,
	int type) {
	cl_uint num_platforms;
	cl_platform_id* platforms;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem bufA, bufB, bufC;
	cl_int err;

	// Platform ID
	err = clGetPlatformIDs(0, NULL, &num_platforms);
	CHECK_ERROR(err);

	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	err = clGetPlatformIDs(num_platforms, platforms, NULL);
	CHECK_ERROR(err);

	// Device ID
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);

	// Create Context
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	// Create Command Queue
	queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
	CHECK_ERROR(err);

	// Create Program Object
	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);

	// Build Program
	char options[1024];
	sprintf(options, "-D TS=%d -D WPT=%d", TS, WPT);
	err = clBuildProgram(program, 1, &device, options, NULL, NULL);
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		char* log;

		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		CHECK_ERROR(err);

		log = (char*)malloc(log_size + 1);
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		CHECK_ERROR(err);

		log[log_size] = '\0';
		printf("Compiler error:\n%s\n", log);
		free(log);
		exit(0);
	};
	CHECK_ERROR(err);

	// Create Kernel
	char kernel_cl[11];
	sprintf(kernel_cl, "vec_mul_%d", type);
	kernel = clCreateKernel(program, kernel_cl, &err);
	CHECK_ERROR(err);

	// Create Buffer
	size_t global_size[2] = { ROW_A, COL_B };
	size_t local_size[2] = { TS, TS };

	if (type == 4) {
		global_size[0] /= WPT;
		local_size[0] /= WPT;
	}
	global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
	global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1] * local_size[1];

	int ROW_B = COL_A;

	bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * ROW_A * COL_A, NULL, &err);
	CHECK_ERROR(err);
	bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * ROW_B * COL_B, NULL, &err);
	CHECK_ERROR(err);
	bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * ROW_A * COL_B, NULL, &err);
	CHECK_ERROR(err);

	// Write Buffer
	err = clEnqueueWriteBuffer(queue, bufA, CL_FALSE, 0, sizeof(float) * ROW_A * COL_A, A, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, bufB, CL_FALSE, 0, sizeof(float) * ROW_B * COL_B, B, 0, NULL, NULL);
	CHECK_ERROR(err);

	// Set Kernel Arg
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 3, sizeof(cl_int), &ROW_A);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 4, sizeof(cl_int), &COL_A);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 5, sizeof(cl_int), &COL_B);
	CHECK_ERROR(err);

	// Execute Kernel
	start = clock();

	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
	CHECK_ERROR(err);

	// Read Buffer
	err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(int) * ROW_A * COL_B, C, 0, NULL, NULL);

	printf("%s\tElapsed time: %f sec\n", kernel_cl, (double)(clock() - start) / CLOCKS_PER_SEC);

	err = clReleaseKernel(kernel);
	CHECK_ERROR(err);
	err = clReleaseProgram(program);
	CHECK_ERROR(err);
	err = clReleaseMemObject(bufA);
	CHECK_ERROR(err);
	err = clReleaseMemObject(bufB);
	CHECK_ERROR(err);
	err = clReleaseMemObject(bufC);
	CHECK_ERROR(err);
	err = clReleaseCommandQueue(queue);
	CHECK_ERROR(err);
	err = clReleaseContext(context);
	CHECK_ERROR(err);

	free(platforms);
}
