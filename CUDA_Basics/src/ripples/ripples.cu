#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cmath>

#include <cpu_anim.h>

#define DIM 1024
#define PI 3.1415926535897932f

__global__ void Kernel(uint8_t* ptr, int32_t tick)
{
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	int32_t offset = x + y * blockDim.x * gridDim.x;

	float fx = x - DIM / 2;
	float fy = y - DIM / 2;
	float d = std::sqrtf(fx * fx + fy * fy);
	uint8_t grey = (uint8_t)(128.0f + 127.0f * std::cos(d / 10.0f - tick / 7.0f) / (d / 10.0f + 1.0f));

	ptr[offset * 4 + 0] = grey;
	ptr[offset * 4 + 1] = grey;
	ptr[offset * 4 + 2] = grey;
	ptr[offset * 4 + 3] = 255;
}

struct DataBlock
{
	uint8_t* d_Bitmap;
	CPUAnimBitmap* Bitmap;
};

void GenerateFrame(DataBlock* d, int32_t tick)
{
	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	Kernel<<<blocks, threads>>>(d->d_Bitmap, tick);
	cudaMemcpy(d->Bitmap->get_ptr(), d->d_Bitmap, d->Bitmap->image_size(), cudaMemcpyDeviceToHost);
}

void Cleanup(DataBlock* d)
{
	cudaFree(d->d_Bitmap);
}

int main()
{
	DataBlock data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);
	data.Bitmap = &bitmap;

	cudaMalloc((void**)&data.d_Bitmap, bitmap.image_size());
	bitmap.anim_and_exit((void(*)(void*, int))GenerateFrame, (void(*)(void*))Cleanup);

	return 0;
}
