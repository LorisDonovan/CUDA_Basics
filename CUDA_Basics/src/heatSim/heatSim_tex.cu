#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "book.h"
#include "cpu_anim.h"

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f

// texture references must be global
texture<float> texConstSrc;
texture<float> texInSrc;
texture<float> texOutSrc;

struct DataBlock
{
	uint8_t* outBitmap;
	float* d_InSrc;
	float* d_OutSrc;
	float* d_ConstSrc;
	CPUAnimBitmap* bitmap;

	cudaEvent_t start, stop;
	float totalTime;
	float frames;
};

__global__ void CopyConstKernel(float* dst)
{
	int32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	int32_t offset = x + y * blockDim.x * gridDim.x;

	float c = tex1Dfetch(texConstSrc, offset);
	if (c != 0) // to preserve previous values
		dst[offset] = c;
}

__global__ void BlendKernel(float* dst, bool dstOut)
{
	int32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	int32_t offset = x + y * blockDim.x * gridDim.x;

	int32_t left = offset - 1;
	int32_t right = offset + 1;
	if (x == 0)
		left++;
	if (x == DIM - 1)
		right--;

	int32_t top = offset - DIM;
	int32_t bottom = offset + DIM;
	if (y == 0)
		top += DIM;
	if (y == DIM - 1)
		bottom -= DIM;

	float t, l, c, r, b;
	if (dstOut) // dstOut indicates whether to use buffer as input or output
	{
		t = tex1Dfetch(texInSrc, top);
		l = tex1Dfetch(texInSrc, left);
		c = tex1Dfetch(texInSrc, offset);
		r = tex1Dfetch(texInSrc, right);
		b = tex1Dfetch(texInSrc, bottom);
	}
	else
	{
		t = tex1Dfetch(texOutSrc, top);
		l = tex1Dfetch(texOutSrc, left);
		c = tex1Dfetch(texOutSrc, offset);
		r = tex1Dfetch(texOutSrc, right);
		b = tex1Dfetch(texOutSrc, bottom);
	}

	dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}

void AnimGPU(DataBlock* d, int ticks)
{
	cudaEventRecord(d->start, nullptr);
	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	CPUAnimBitmap* bitmap = d->bitmap;

	volatile bool dstOut = true;
	for (int i = 0; i < 90; i++)
	{
		float* inSrc, * outSrc;
		if (dstOut)
		{
			inSrc = d->d_InSrc;
			outSrc = d->d_OutSrc;
		}
		else
		{
			outSrc = d->d_InSrc;
			inSrc = d->d_OutSrc;
		}

		CopyConstKernel << <blocks, threads >> > (inSrc);
		BlendKernel << <blocks, threads >> > (outSrc, dstOut);
		dstOut = !dstOut;
	}

	float_to_color << <blocks, threads >> > (d->outBitmap, d->d_InSrc);
	cudaMemcpy(bitmap->get_ptr(), d->outBitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);

	cudaEventRecord(d->stop, nullptr);
	cudaEventSynchronize(d->stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, d->start, d->stop);
	d->totalTime += elapsedTime;
	d->frames++;
	printf("\rAverage Time per frame: %5.1f ms", d->totalTime / d->frames);
}

void AnimExit(DataBlock* d)
{
	cudaUnbindTexture(texInSrc);
	cudaUnbindTexture(texOutSrc);
	cudaUnbindTexture(texConstSrc);

	cudaFree(d->d_InSrc);
	cudaFree(d->d_OutSrc);
	cudaFree(d->d_ConstSrc);

	cudaEventDestroy(d->start);
	cudaEventDestroy(d->stop);
}

int main()
{
	DataBlock data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);
	data.bitmap = &bitmap;
	data.totalTime = 0;
	data.frames = 0;

	cudaEventCreate(&data.start);
	cudaEventCreate(&data.stop);

	size_t imageSize = bitmap.image_size();

	cudaMalloc((void**)&data.outBitmap,  imageSize);
	// assume float == 4 chars in size (i.e., rgba)
	cudaMalloc((void**)&data.d_InSrc,    imageSize);
	cudaMalloc((void**)&data.d_OutSrc,   imageSize);
	cudaMalloc((void**)&data.d_ConstSrc, imageSize);

	// bind texture references to buffers
	cudaBindTexture(NULL, texConstSrc, data.d_ConstSrc, imageSize);
	cudaBindTexture(NULL, texInSrc,    data.d_InSrc,    imageSize);
	cudaBindTexture(NULL, texOutSrc,   data.d_OutSrc,   imageSize);

	// initialize the constant data
	auto temp = (float*)malloc(imageSize);
	for (int i = 0; i < DIM * DIM; i++)
	{
		temp[i] = 0;
		int32_t x = i % DIM;
		int32_t y = i / DIM;
		if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
			temp[i] = MAX_TEMP;
	}
	temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
	temp[DIM * 700 + 100] = MIN_TEMP;
	temp[DIM * 300 + 300] = MIN_TEMP;
	temp[DIM * 200 + 700] = MIN_TEMP;
	for (int y = 800; y < 900; y++)
	{
		for (int x = 400; x < 500; x++)
			temp[x + y * DIM] = MIN_TEMP;
	}
	cudaMemcpy(data.d_ConstSrc, temp, imageSize, cudaMemcpyHostToDevice);

	// initialize the input data
	for (int y = 800; y < DIM; y++)
	{
		for (int x = 0; x < 200; x++)
			temp[x + y * DIM] = MAX_TEMP;
	}
	cudaMemcpy(data.d_InSrc, temp, imageSize, cudaMemcpyHostToDevice);
	free(temp);

	bitmap.anim_and_exit((void (*)(void*, int))AnimGPU, (void (*)(void*))AnimExit);

	return 0;
}
