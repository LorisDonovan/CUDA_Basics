#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU
__global__ void addVector(int32_t N, float* x, float* y, float* result);

void Print(float* x, int32_t n);

int main()
{
	int32_t id = cudaGetDevice(&id);

	const int32_t N = 1 << 20; // 2^20
	size_t bytes = N * sizeof(float);

	float* x, * y, * result;
	cudaMallocManaged(&x, bytes);
	cudaMallocManaged(&y, bytes);
	cudaMallocManaged(&result, bytes);

	// initialize array
	for (int i = 0; i < N; i++)
	{
		x[i] = i * 2.0f;
		y[i] = i * 3.0f;
	}

	int32_t blockSize = 256;
	int32_t numBlocks = (N + blockSize - 1) / blockSize;

	// prefetch to the gpu
	cudaMemPrefetchAsync(x, bytes, id);
	cudaMemPrefetchAsync(y, bytes, id);
	addVector<<<numBlocks, blockSize>>>(N, x, y, result);
	// <<<no. of thread blocks, no. of threads in a block>>>
	// because we passed only int this is only 1D 

	cudaDeviceSynchronize(); // to make cpu wait for the kernel to complete its operation

	cudaMemPrefetchAsync(result, bytes, cudaCpuDeviceId); // prefetch to cpu

	std::cout << "result:" << std::endl;
	Print(result, 5);

	// deallocating memory
	cudaFree(x);
	cudaFree(y);
	cudaFree(result);

	std::cin.get();
	return 0;
}

__global__ void addVector(int32_t N, float* x, float* y, float* result)
{
	int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < N)
		result[tid] = x[tid] + y[tid];
}

void Print(float* x, int32_t n)
{
	for (int i = 0; i < n; i++)
		std::cout << x[i] << std::endl;
}
