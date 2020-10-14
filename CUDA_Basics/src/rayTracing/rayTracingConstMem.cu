#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <random>
#include <functional>
#include <iomanip>

#include "cpu_bitmap.h"

#define DIM 1024
#define INF 2e10f
#define NUM_SPHERES 20

struct Sphere
{
	float r, g, b;
	float x, y, z;
	float radius;

	__device__ float Hit(float ox, float oy, float* n) // pixel (ox, oy)
	{
		float dx = ox - x;
		float dy = oy - y;

		if ((dx * dx + dy * dy) < (radius * radius))
		{
			float dz = std::sqrtf(radius * radius - dx * dx - dy * dy);
			*n = dz / radius;
			return dz + z;
		}
		return -INF;
	}
};

__constant__ Sphere s[NUM_SPHERES];

__global__ void Kernel(uint8_t* ptr);
void InitSphere(Sphere* spheres, std::function<float(float)> randFunc);

int main()
{
	std::mt19937 gen(0);
	std::uniform_real_distribution<float> distribution(0, 1);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	CPUBitmap bitmap(DIM, DIM);
	uint8_t* d_Bitmap;

	cudaMalloc(&d_Bitmap, bitmap.image_size());

	Sphere* temp_s = (Sphere*)malloc(NUM_SPHERES * sizeof(Sphere));
	InitSphere(temp_s, [&](float x) { return x * distribution(gen); });
	cudaMemcpyToSymbol(s, temp_s, NUM_SPHERES * sizeof(Sphere));
	free(temp_s);

	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	Kernel<<<grids, threads>>>(d_Bitmap);
	cudaMemcpy(bitmap.get_ptr(), d_Bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);

	// get stop time, and display the timing results
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to generate:  %3.1f ms\n", elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	bitmap.display_and_exit();

	cudaFree(d_Bitmap);
	return 0;
}

__global__ void Kernel(uint8_t* ptr)
{
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	int32_t offset = x + y * blockDim.x * gridDim.x;

	float ox = x - DIM / 2;
	float oy = y - DIM / 2;
	float r = 0, g = 0, b = 0;
	float maxz = -INF;

	for (int i = 0; i < NUM_SPHERES; i++)
	{
		float n;
		float t = s[i].Hit(ox, oy, &n);
		if (t > maxz)
		{
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
			maxz = t;
		}
	}

	ptr[offset * 4 + 0] = (int32_t)(r * 255);
	ptr[offset * 4 + 1] = (int32_t)(g * 255);
	ptr[offset * 4 + 2] = (int32_t)(b * 255);
	ptr[offset * 4 + 3] = 255;
}

void InitSphere(Sphere* spheres, std::function<float(float)> randFunc)
{
	for (int i = 0; i < NUM_SPHERES; i++)
	{
		spheres[i].r = randFunc(1.0f);
		spheres[i].g = randFunc(1.0f);
		spheres[i].b = randFunc(1.0f);
		spheres[i].x = randFunc(1000.0f) - 500.0f;
		spheres[i].y = randFunc(1000.0f) - 500.0f;
		spheres[i].z = randFunc(1000.0f) - 500.0f;
		spheres[i].radius = randFunc(100.0f) + 20.0f;
	}
}
