#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>
#include <random>
#include <cassert>
#include <iostream>
#include <algorithm>


__global__ void CopyMatrix(float* dest, const float* src, int32_t N);
__global__ void Transpose(float* mat, float* matT, int32_t N);
__global__ void MatrixMult(float* a, float* b, float* c, int32_t N); 

void VerifyResult(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, int32_t N);

int main()
{
	const int32_t N = 1 << 10; // 1024 x 1024
	size_t bytes = N * N * sizeof(float);

	std::mt19937 gen1(0);
	std::mt19937 gen2(1);
	std::uniform_int_distribution<int> distribution(0, 10);

	std::vector<float> h_A(N * N);
	std::vector<float> h_B(N * N);
	std::vector<float> h_C(N * N);
	std::generate(h_A.begin(), h_A.end(), [&]() { return distribution(gen1); });
	std::generate(h_B.begin(), h_B.end(), [&]() { return distribution(gen2); });

	float* d_A, * d_At, * d_B, * d_C;
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_At, bytes);
	cudaMalloc(&d_B, bytes);
	cudaMalloc(&d_C, bytes);

	cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

	int32_t blockSize = 16; // n x n
	int32_t numBlocks = (N + blockSize - 1) / blockSize;

	dim3 grid(numBlocks, numBlocks);
	dim3 threads(blockSize, blockSize);

	std::cout << "Matrix GPU operation started..." << std::endl;
	Transpose<<<grid, threads>>>(d_A, d_At, N);
	MatrixMult<<<grid, threads>>>(d_At, d_B, d_C, N);
	std::cout << "Completed Matrix operation!" << std::endl;
	cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

	std::cout << "Matrix verification started..." << std::endl;
	VerifyResult(h_A, h_B, h_C, N);
	std::cout << "Matrix Verified!" << std::endl;

	cudaFree(d_A);
	cudaFree(d_At);
	cudaFree(d_B);
	cudaFree(d_C);

	std::cin.get();
	return 0;
}

__global__ void CopyMatrix(float* dest, const float* src, int32_t N)
{
	int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
	int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < N && col < N)
		dest[row * N + col] = src[row * N + col];
}

__global__ void Transpose(float* mat, float* matT, int32_t N)
{
	int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
	int32_t col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N && col < N)
		matT[col * N + row] = mat[row * N + col];
}

__global__ void MatrixMult(float* a, float* b, float* c, int32_t N)
{
	int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
	int32_t col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N && col < N)
	{
		c[row * N + col] = 0;
		for (int i = 0; i < N; i++)
			c[row * N + col] += a[i * N + row] * b[i * N + col];
	}
}

void VerifyResult(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, int32_t N)
{
	float temp;
	for (int row = 0; row < N; row++)
	{
		for (int col = 0; col < N; col++)
		{
			temp = 0;
			for (int i = 0; i < N; i++)
				temp += a[row * N + i] * b[i * N + col];
			assert(c[row * N + col] == temp);
		}
	}

}
