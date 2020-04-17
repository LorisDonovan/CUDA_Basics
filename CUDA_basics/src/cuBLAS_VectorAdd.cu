// explicitly link the libraries

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>

void VerifyResult(std::vector<float> &a, std::vector<float> &b, std::vector<float> &c, float factor, int n);

int main()
{
	int N = 1 << 16; // 2 ^ 16
	size_t bytes = N * sizeof(float);

	std::vector<float> h_a(N);
	std::vector<float> h_b(N);
	std::vector<float> h_c(N);

	std::generate(h_a.begin(), h_a.end(), []() { return std::rand() % 10; });
	std::generate(h_b.begin(), h_b.end(), []() { return std::rand() % 10; });

	float *d_a, *d_b;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);

	// Initialize and create new context
	cublasHandle_t handle;
	cublasCreate_v2(&handle);

	// Copy the vectors over to the device
	cublasSetVector(N, sizeof(float), h_a.data(), 1, d_a, 1);
	cublasSetVector(N, sizeof(float), h_b.data(), 1, d_b, 1);

	// Launch simple saxpy kernel (single precision a * x + y)
	const float scale = 2.0f;
	cublasSaxpy(handle, N, &scale, d_a, 1, d_b, 1);

	// Copy the result vector back out
	cublasGetVector(N, sizeof(float), d_b, 1, h_c.data(), 1);

	VerifyResult(h_a, h_b, h_c, scale, N);
	std::cout << "Completed Successfully!" << std::endl;

	cublasDestroy(handle);
	cudaFree(d_a);
	cudaFree(d_b);

	std::cin.get();
	return 0;
}

void VerifyResult(std::vector<float> &a, std::vector<float> &b, std::vector<float> &c, float factor, int n)
{
	for (int i = 0; i < n; i++)
	{
		assert(c[i] == factor * a[i] + b[i]);
	}
}
