#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "cpu_bitmap.h"

#define DIM 1024
#define PI 3.1415926535897932f

__global__ void Kernel(uint8_t* ptr)
{
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	int32_t offset = x + y * blockDim.x * gridDim.x;

	const float period = 128.0f;
	__shared__ float shared[16][16];

	shared[threadIdx.x][threadIdx.y] = 255 * (std::sinf(x * 2.0f * PI / period) + 1.0f) * (std::sinf(y * 2.0f * PI / period) + 1.0f) / 4.0f;
	
	__syncthreads(); // comment this to see how this kernel behaves when threads are not in sync
	
	ptr[offset * 4 + 0] = 0;
	ptr[offset * 4 + 1] = shared[15 - threadIdx.x][15 - threadIdx.y];
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
}

int main()
{
	CPUBitmap bitmap(DIM, DIM);
	uint8_t* d_Bitmap;

	cudaMalloc(&d_Bitmap, bitmap.image_size());

	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	Kernel<<<grids, threads>>>(d_Bitmap);

	cudaMemcpy(bitmap.get_ptr(), d_Bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
	bitmap.display_and_exit();
	cudaFree(d_Bitmap);
	return 0;
}
