#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <functional>
#include <random>

#define MIN(a, b) (a < b ? a : b)
#define SUM_SQUARES(x) (x*(x+1)*(2*x+1)/6)

constexpr int32_t N = 1 << 10;
const int32_t threadsPerBlock = 256;
const int32_t blocksPerGrid = MIN(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void DotProduct(float* a, float* b, float* c)
{
	__shared__ float cache[threadsPerBlock];
	int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t cacheIdx = threadIdx.x;

	float temp = 0;
	while (tid < N)
	{
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}
	cache[cacheIdx] = temp;
	__syncthreads();
	// reduction; threadsPerBlock must be power of 2
	int32_t i = blockDim.x / 2;
	while (i != 0)
	{
		if (cacheIdx < i)
			cache[cacheIdx] += cache[cacheIdx + i];
		__syncthreads();
		i /= 2;
	}
	if (cacheIdx == 0)
		c[blockIdx.x] = cache[0]; // at the end of the operation the result is accumulated at the 0th index of cache
	// the last operation is done in the CPU 
}

int main()
{
	float* a = (float*)malloc(N * sizeof(float));
	float* b = (float*)malloc(N * sizeof(float));
	float* partial_c = (float*)malloc(blocksPerGrid * sizeof(float));
	float c = 0;

	float* d_a, * d_b, *d_c;
	cudaMalloc(&d_a, N * sizeof(float));
	cudaMalloc(&d_b, N * sizeof(float));
	cudaMalloc(&d_c, blocksPerGrid * sizeof(float));

	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i * 2;
	}

	cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

	DotProduct<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);

	cudaMemcpy(partial_c, d_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

	// finishing the last step in CPU
	// the result is too small for a GPU
	for (int i = 0; i < blocksPerGrid; i++)
		c += partial_c[i];
	std::cout << "actual result   = " << 2 * SUM_SQUARES((float)(N - 1)) << std::endl;
	std::cout << "result obtained = " << c << std::endl;

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(a);
	free(b);
	free(partial_c);

	std::cin.get();
	return 0;
}
