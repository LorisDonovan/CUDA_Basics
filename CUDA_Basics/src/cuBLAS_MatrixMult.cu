#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h> // for generating random numbers on the gpu

#include <iostream>
#include <vector>
#include <ctime>
#include <cassert>

// cuBLAS assumes column major

void VerifyResult(float *a, float *b, float *c, int M, int N, int K);

int main()
{
	// Dimensions for our matrices
	const int M = 1 << 9;
	const int N = 1 << 8;
	const int K = 1 << 7;

	const size_t bytes_a = M * K * sizeof(float);
	const size_t bytes_b = K * N * sizeof(float);
	const size_t bytes_c = M * N * sizeof(float);

	// Vectors for host data
	std::vector<float> h_a(M * K);
	std::vector<float> h_b(K * N);
	std::vector<float> h_c(M * N);

	// Allocate device memory
	float *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, bytes_a);
	cudaMalloc(&d_b, bytes_b);
	cudaMalloc(&d_c, bytes_c);

	// Pseudo random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	// set seed
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

	// Fill the matrix with random numbers on the device
	curandGenerateUniform(prng, d_a, M * K);
	curandGenerateUniform(prng, d_a, K * N);

	// cuBLAS handle
	cublasHandle_t handle;
	cublasCreate(&handle);

	float alpha = 1.0f;
	float beta = 0.0f;

	// Calculate: c = (alpha * a) * b + (beta * c)
	// MxN = MxK * KxN 
	// Signature: handle, operation, operation,   M, N, K, alpha,   A, lda, B, ldb, beta, C, ldc
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_a, M, d_b, K, &beta, d_c, M);

	// copy back to host 
	cudaMemcpy(h_a.data(), d_a, bytes_a, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b.data(), d_b, bytes_b, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);

	VerifyResult(h_a.data(), h_b.data(), h_c.data(), M, N, K);
	std::cout << "Completed Successfully!" << std::endl;

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	std::cin.get();
	return 0;
}

void VerifyResult(float *a, float *b, float *c, int M, int N, int K)
{
	float epsilon = 0.001; // Tolerance for our result
	for (int row = 0; row < M; row++)
	{
		for (int col = 0; col < N; col++)
		{
			float temp = 0.0f;
			for (int i = 0; i < K; i++)
			{
				temp += a[row + M * i] * b[col * K + i];	// assuming col maj
			}
			assert(std::fabs(c[col * M + row] - temp) <= epsilon);
		}
	}
}
