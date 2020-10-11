#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

#include <iostream>
#include <cassert>
#include <algorithm>
#include <random>
#include <functional>

template<typename T>
void VectorInit(T* a, const int32_t N, std::function<T()> randFunc);
void VerifyResult(float* a, float* b, float* c, float factor, const int32_t N);

int main()
{
	constexpr int32_t N = 1 << 16;
	size_t bytes = N * sizeof(float);

	std::mt19937 gen1(0);
	std::mt19937 gen2(1);
	std::uniform_int_distribution<int> distribution(0, 10); // [0, 10]

	float* h_A = (float*)malloc(bytes);
	float* h_B = (float*)malloc(bytes);
	float* h_C = (float*)malloc(bytes);

	VectorInit<float>(h_A, N, [&]() { return distribution(gen1); });
	VectorInit<float>(h_B, N, [&]() { return distribution(gen2); });

	float* d_A, * d_B;
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);

	// Initialize and create new context
	cublasHandle_t handle;
	cublasCreate_v2(&handle);

	// Copy the vectors over to the device
	cublasSetVector(N, sizeof(float), h_A, 1, d_A, 1);
	cublasSetVector(N, sizeof(float), h_B, 1, d_B, 1);

	// Launch simple saxpy kernel (single precision a * x + y)
	// result stored in y
	const float scale = 2.0f;
	cublasSaxpy(handle, N, &scale, d_A, 1, d_B, 1);

	// Copy the result vector back out
	cublasGetVector(N, sizeof(float), d_B, 1, h_C, 1);

	VerifyResult(h_A, h_B, h_C, scale, N);
	std::cout << "Vector verified!" << std::endl;

	cublasDestroy_v2(handle);
	cudaFree(d_A);
	cudaFree(d_B);
	free(h_A);
	free(h_B);
	free(h_C);

	std::cin.get();
	return 0;
}

template<typename T>
void VectorInit(T* a, const int32_t N, std::function<T()> randFunc)
{
	for (int i = 0; i < N; i++)
		a[i] = randFunc();
}

void VerifyResult(float* a, float* b, float* c, float factor, const int32_t N)
{
	for (int i = 0; i < N; i++)
	{
		assert(c[i] == factor * a[i] + b[i]);
	}
}
