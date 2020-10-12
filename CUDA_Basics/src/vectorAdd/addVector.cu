#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cassert>

// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU
__global__ void addVector(int32_t N, float* x, float* y);
__global__ void addVector_v2(int32_t N, float* x, float* y); // vector addition for arbitrary length N

void Print(float* x, int32_t n);
void VerifyResult(float* a, float* b, float* c, int32_t N);

int main()
{
	constexpr int32_t N = 1 << 20; // 2^n
	size_t bytes = N * sizeof(float);

	// allocating host memory
	float* h_A = (float*)malloc(bytes);
	float* h_B = (float*)malloc(bytes);
	float* h_C = (float*)malloc(bytes);

	float* d_A, * d_B;
	// allocating device memory
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);

	// initialize array
	for (int i = 0; i < N; i++)
	{
		h_A[i] = i * 2.0f;
		h_B[i] = i * 3.0f;
	}

	// copying from host to device
	cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

	int32_t blockSize = 256;
	int32_t numBlocks = (N + blockSize - 1) / blockSize;

	addVector<<<numBlocks, blockSize>>>(N, d_A, d_B);
	// <<<no. of thread blocks, no. of threads in a block>>>
	// because we passed only int this is only 1D 

	//addVector_v2<<<256, 256>>>(N, d_A, d_B);

	cudaMemcpy(h_C, d_B, bytes, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize(); // to make cpu wait for the kernel to complete its operation

	VerifyResult(h_A, h_B, h_C, N);
	std::cout << "Vector Verified!" << std::endl;

	// deallocating memory
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);

	std::cin.get();
	return 0;
}

__global__ void addVector(int32_t N, float* x, float* y)
{
	int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < N)
		y[tid] = x[tid] + y[tid];
}

__global__ void addVector_v2(int32_t N, float* x, float* y)
{
	int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	while (tid < N)
	{
		y[tid] = x[tid] + y[tid];
		tid += blockDim.x * gridDim.x;
	}
}

void VerifyResult(float* a, float* b, float* c, int32_t N)
{
	for (int i = 0; i < N; i++)
		assert(c[i] == a[i] + b[i]);
}

void Print(float* x, int32_t n)
{
	for (int i = 0; i < n; i++)
		std::cout << x[i] << std::endl;
}
