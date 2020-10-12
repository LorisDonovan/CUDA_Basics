#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include <cpu_bitmap.h>

#define DIM 1000

struct Complex
{
	float real;
	float img;

	__device__ Complex(float x, float y)
		: real(x), img(y) {}

	__device__ float Magnitude2() { return real * real + img * img; }

	__device__ Complex operator*(const Complex& other)
	{
		return Complex((real * other.real - img * other.img), real * other.img + img * other.real);
	}

	__device__ Complex operator+(const Complex& other)
	{
		return Complex(real + other.real, img + other.img);
	}
};

__device__ int32_t Julia(int32_t x, int32_t y)
{
	const float scale = 1.5f;
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	Complex c(-0.8f, 0.156f);
	Complex z(jx, jy);

	for (int i = 0; i < 200; i++)
	{
		z = z * z + c;
		if (z.Magnitude2() > 1000)
			return 0;
	}
	return 1;
}

__global__ void Kernel(uint8_t* ptr)
{
	int32_t x = blockIdx.x;
	int32_t y = blockIdx.y;

	int32_t offset = x + y * DIM;

	int32_t juliaValue = Julia(x, y);
	ptr[offset * 4 + 0] = 50 * juliaValue; // R
	ptr[offset * 4 + 1] = 145 * juliaValue; // G
	ptr[offset * 4 + 2] = 168 * juliaValue; // B
	ptr[offset * 4 + 3] = 255; // A
}

int main()
{
	CPUBitmap bitmap(DIM, DIM);
	uint8_t* d_Bitmap;
	cudaMalloc(&d_Bitmap, bitmap.image_size());


	dim3 grid(DIM, DIM);
	Kernel<<<grid, 1>>>(d_Bitmap);

	cudaMemcpy(bitmap.get_ptr(), d_Bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
	bitmap.display_and_exit();

	cudaFree(d_Bitmap);
	return 0;
}