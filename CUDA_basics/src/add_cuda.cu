#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cmath>

__global__ void add(int n, float* x, float* y)
{
	int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (tid < n)
		y[tid] = x[tid] + y[tid];
}

int main()
{
	int N = 1 << 20;

	float *x, *y;
	cudaMallocManaged(&x, N * sizeof(float));	// similar to new
	cudaMallocManaged(&y, N * sizeof(float));

	// initialize the array
	for (int i = 0; i < N; i++)
	{
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;

	// lauching gpu thread
	add <<< numBlocks, blockSize >>> (N, x, y);	//<<<>>> for cuda kernel launch
	// <<<no. of thread blocks , no. of threads in a block>>>

	// to make cpu wait for the kernel to complete its operation
	cudaDeviceSynchronize();

	// check for errors ( all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	std::cout << "Max Error: " << maxError << std::endl;
	cudaDeviceSynchronize();

	cudaFree(x);	// similar to delete
	cudaFree(y);

	std::cin.get();
	return 0;
}
