#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <functional>

const int bits = 7;
const int N = 1 << bits; // Matrix size of (2^n) x (2^n)
const int SHMEM_SIZE = 1 << 10;	// shared memory tile size (blockDim x blockDim)

__global__ void TileMatrixMult(int *a, int *b, int *c);
__global__ void MatrixMult(int *a, int *b, int *c);
void VerifyResult(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c);

int main()
{
	size_t bytes = N * N * sizeof(int); // size (in bytes) of matrix

	// Host vectors
	std::vector<int> h_a(N * N);
	std::vector<int> h_b(N * N);
	std::vector<int> h_c(N * N);

	// Initialize matrices
	std::generate(h_a.begin(), h_a.end(), []() {return rand() % 10; });
	std::generate(h_b.begin(), h_b.end(), []() {return rand() % 10; });

	// Device pointers
	int *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// Copy data to the device
	cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
	
	int THREADS = 32; // Threads per CTA dimension
	int BLOCKS = N / THREADS; // Blocks per grid dimension (assumes THREADS divides N evenly)

	// Use dim3 structs for block  and grid dimensions
	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);
	
	// launch kernel
	//TileMatrixMult <<<blocks, threads>>> (d_a, d_b, d_c);
	MatrixMult <<<blocks, threads>>> (d_a, d_b, d_c);
	
	cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost); // Copy back to the host
	
	VerifyResult(h_a, h_b, h_c); // Check result
	std::cout << "Completed Successfully!" << std::endl;

	std::cin.get();
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}

// Cache tiling method
__global__ void TileMatrixMult(int *a, int *b, int *c)
{
	// two statically sized pieces of shared memory
	__shared__ int A[SHMEM_SIZE];
	__shared__ int B[SHMEM_SIZE];

	// Calculating global row and col
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// instead of going through the entire matrix, we do it based upon
	// the (total size of the matrix) / (tileSize)
	int temp = 0;
	for (int i = 0; i < N; i += blockDim.x)
	{
		// Load elements for this tile
		A[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];
		B[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];

		// Ensure all threads have loaded their data before processing
		__syncthreads();

		// Do matrix multiplication on the small matrix
		for (int j = 0; j < blockDim.x; j++)
		{
			temp += A[threadIdx.y * blockDim.x + j] * B[j * blockDim.x + threadIdx.x];
		}
		// Ensure some threads don't progress and stomp current shared memory values
		__syncthreads();
	}
	c[(row * N) + col] = temp;
}

__global__ void MatrixMult(int *a, int *b, int *c)
{
	// Compute each thread's row and col
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Iterate over row and down col
	c[row * N + col] = 0;
	for (int k = 0; k < N; k++)
	{
		// Accumulate result for a single element
		c[row * N + col] += a[row * N + k] * b[k * N + col];
	}
}

void VerifyResult(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c)
{
	// for every row
	for (int i = 0; i < N; i++)
	{
		// for every col
		for (int j = 0; j < N; j++)
		{
			// for every element
			int temp = 0;
			for (int k = 0; k < N; k++)
			{
				temp += a[i * N + k] * b[k * N + j];
			}
			//std::printf("%3d", temp);
			//std::printf("%3d", c[i * N + j]);
			assert(temp == c[i * N + j]);
		}
		//std::printf("\n");
	}
}