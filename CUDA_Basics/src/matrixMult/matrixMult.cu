#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>
#include <random>
#include <cassert>
#include <iostream>
#include <algorithm>


const int32_t SHMEM_SIZE = 16 * 16; // tile size

__global__ void MatrixMult(float* a, float* b, float* c, int32_t N); // noramal 
__global__ void TiledMatrixMult(float* a, float* b, float* c, int32_t N); // cache tiling
void VerifyResult(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, int32_t N);

int main()
{
	const int32_t N = 1 << 10; // 1024 x 1024
	size_t bytes = N * N * sizeof(float);

	std::mt19937 gen1(0);
	std::mt19937 gen2(1);
	std::uniform_int_distribution<int> distribution(0, 10);

	std::vector<float> h_A(N*N);
	std::vector<float> h_B(N*N);
	std::vector<float> h_C(N*N);
	std::generate(h_A.begin(), h_A.end(), [&]() { return distribution(gen1); });
	std::generate(h_B.begin(), h_B.end(), [&]() { return distribution(gen2); });

	float* d_A, * d_B, * d_C;
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);
	cudaMalloc(&d_C, bytes);

	cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

	int32_t blockSize = 16; // 16x16
	int32_t numBlocks = (N + blockSize - 1) / blockSize;

	dim3 grid(numBlocks, numBlocks);
	dim3 threads(blockSize, blockSize);

	std::cout << "Matrix Multiplication GPU started..." << std::endl;
	//TiledMatrixMult<<<grid, threads>>>(d_A, d_B, d_C, N);
	MatrixMult<<<grid, threads>>>(d_A, d_B, d_C, N);
	cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
	std::cout << "Completed Matrix Multiplication!" << std::endl;

	std::cout << "Matrix Multiplication verification started..." << std::endl;
	VerifyResult(h_A, h_B, h_C, N);
	std::cout << "Matrix Multiplication Verified!" << std::endl;

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	std::cin.get();
	return 0;
}

__global__ void MatrixMult(float* a, float* b, float* c, int32_t N)
{
	int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
	int32_t col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N && col < N)
	{
		c[row * N + col] = 0;
		for (int i = 0; i < N; i++)
			c[row * N + col] += a[row * N + i] * b[i * N + col];
	}
}

__global__ void TiledMatrixMult(float* a, float* b, float* c, int32_t N) // cache tiling
{
	int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
	int32_t col = blockIdx.x * blockDim.x + threadIdx.x;

	// Statically allocated shared memory
	__shared__ int32_t s_A[SHMEM_SIZE];
	__shared__ int32_t s_B[SHMEM_SIZE];

	if (row < N && col < N)
	{
		float temp = 0;
		// Sweep tile across matrix
		for (int i = 0; i < N; i += blockDim.x)
		{
			// Load elements for this tile
			s_A[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];
			s_B[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];

			// Ensure all threads have loaded their data before processing
			__syncthreads();

			// Do matrix multiplication on the small matrix
			for (int j = 0; j < blockDim.x; j++)
			{
				temp += s_A[threadIdx.y * blockDim.x + j] * s_B[j * blockDim.x + threadIdx.x];
			}
			// Ensure some threads don't progress and stomp current shared memory values
			__syncthreads();
		}
		c[(row * N) + col] = temp;
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
