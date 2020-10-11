#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>

#include <iostream>
#include <cassert>

// cuBLAS operates on matrix in col major

void VerifyResult(float* a, float* b, float* c, const int32_t N);
void VerifyResult(float* a, float* b, float* c, const int32_t M, const int32_t N, const int32_t K);

int main()
{
	constexpr int32_t N = 1 << 10;
	size_t bytes = N * N * sizeof(float);

	float* h_A = (float*)malloc(bytes);
	float* h_B = (float*)malloc(bytes);
	float* h_C = (float*)malloc(bytes);

	float* d_A, * d_B, * d_C;
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);
	cudaMalloc(&d_C, bytes);

	// Pseudo Random Number Generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_MT19937);

	// fill matrix with random values in device
	curandGenerateUniform(prng, d_A, N * N);
	curandGenerateUniform(prng, d_B, N * N);

	cublasHandle_t handle;
	cublasCreate_v2(&handle);

	const float alpha = 1.0f;
	const float beta  = 0.0f;

	// Calculate: c = (alpha * a) * b + (beta * c)
	// MxN = MxK * KxN 
	// Signature: handle, operation, operation,   M, N, K, alpha,   A, lda, B, ldb, beta, C, ldc
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

	// copy back to host 
	cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

	VerifyResult(h_A, h_B, h_C, N);
	std::cout << "Completed Successfully!" << std::endl;

	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cublasDestroy_v2(handle);

	std::cin.get();
	return 0;
}

void VerifyResult(float* a, float* b, float* c, const int32_t N)
{
	float temp;
	const float epsilon = 0.001;
	for (int row = 0; row < N; row++)
	{
		for (int col = 0; col < N; col++)
		{
			temp = 0;
			for (int i = 0; i < N; i++)
				temp += a[i * N + row] * b[col * N + i]; // col major
			assert(std::fabs(c[col * N + row] - temp) <= epsilon);
		}
	}
}


void VerifyResult(float* a, float* b, float* c, const int32_t M, const int32_t N, const int32_t K)
{
	float epsilon = 0.001; // Tolerance for our result
	for (int row = 0; row < M; row++)
	{
		for (int col = 0; col < N; col++)
		{
			float temp = 0.0f;
			for (int i = 0; i < K; i++)
				temp += a[i * M + row] * b[col * K + i];	// assuming col maj
			assert(std::fabs(c[col * M + row] - temp) <= epsilon);
		}
	}
}
