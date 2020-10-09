#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU
__global__ void addVector(int32_t N, float* x, float* y);

void Print(float* x, int32_t n);

int main()
{
	const int32_t N = 1 << 20; // 2^20
	size_t bytes = N * sizeof(float);

	// allocating host memory
	float* h_X = (float*)malloc(bytes);
	float* h_Y = (float*)malloc(bytes);

	float* d_X, * d_Y;
	// allocating device memory
	cudaMalloc(&d_X, bytes);
	cudaMalloc(&d_Y, bytes);

	// initialize array
	for (int i = 0; i < N; i++)
	{
		h_X[i] = i * 2.0f;
		h_Y[i] = i * 3.0f;
	}

	// copying from host to device
	cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, h_Y, bytes, cudaMemcpyHostToDevice);

	int32_t blockSize = 256;
	int32_t numBlocks = (N + blockSize - 1) / blockSize;

	addVector<<<numBlocks, blockSize>>>(N, d_X, d_Y);
	// <<<no. of thread blocks, no. of threads in a block>>>
	// because we passed only int this is only 1D 

	cudaMemcpy(h_Y, d_Y, bytes, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize(); // to make cpu wait for the kernel to complete its operation

	std::cout << "result:" << std::endl;
	Print(h_Y, 5);

	// deallocating memory
	cudaFree(h_X);
	cudaFree(h_Y);
	cudaFree(d_X);
	cudaFree(d_Y);

	std::cin.get();
	return 0;
}

__global__ void addVector(int32_t N, float* x, float* y)
{
	int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < N)
		y[tid] = x[tid] + y[tid];
}

void Print(float* x, int32_t n)
{
	for (int i = 0; i < n; i++)
		std::cout << x[i] << std::endl;
}
