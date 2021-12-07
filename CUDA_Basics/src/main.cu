#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define cudaCheckError(val) CheckCuda(val, #val, __FILE__, __LINE__)


void CheckCuda(cudaError_t result, const char* func, const char* file, const int32_t line)
{
	if (result)
	{
		printf("CUDA::ERROR::%d - function:'%s' - file: '%s' - line: %d\n\
			%s\n", result, func, file, line, cudaGetErrorString(result));
		 __debugbreak();
		//exit(-1);
	}
}

__global__ void Kernel(uint8_t* outImg, int32_t width, int32_t height, int32_t channels)
{
	int32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	int32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= width || y >= height)
		return;

	int32_t idx = (x + y * width) * channels;

	float r = float(x) / float(width);
	float g = float(y) / float(height);
	float b = 0.2f;

	outImg[idx + 0] = uint8_t(r * 255);
	outImg[idx + 1] = uint8_t(g * 255);
	outImg[idx + 2] = uint8_t(b * 255);
	outImg[idx + 3] = 255;
}


int main(int argc, char** argv)
{
	constexpr float aspectRatio = 16.0f / 9.0f;
	constexpr uint32_t height   = 270;
	constexpr uint32_t width    = static_cast<uint32_t>(height * aspectRatio);
	constexpr uint32_t channels = 4; // number of channels in the image

	constexpr int32_t tx = 8;
	constexpr int32_t ty = 8;

	const int32_t imgSize = width * height * channels;
	const size_t  bytes   = imgSize * sizeof(uint8_t);

	uint8_t* img = (uint8_t*)malloc(bytes);
	memset(img, 255, imgSize);
	uint8_t* d_Img;
	cudaCheckError(cudaMalloc((void**)&d_Img, bytes));

	dim3 threads(tx, ty);
	dim3 blocks((width + tx - 1) / tx, (height + ty - 1) / ty);
	Kernel<<<threads, blocks>>>(d_Img, width, height, channels);
	cudaCheckError(cudaGetLastError());
	cudaCheckError(cudaDeviceSynchronize());

	cudaCheckError(cudaMemcpy(img, d_Img, bytes, cudaMemcpyDeviceToHost));

	// for (int y = 0; y < height; y++)
	// {
	// 	for (int x = 0; x < width; x++)
	// 	{
	// 		// printf("%4d ", idx);
	// 		// idx += 4;
	// 		int32_t idx = (x + y * width) * 4;

	// 		float r = float(x) / float(width);
	// 		float g = float(y) / float(height);
	// 		float b = 0.2f;

	// 		img[idx + 0] = uint8_t(r * 255);
	// 		img[idx + 1] = uint8_t(g * 255);
	// 		img[idx + 2] = uint8_t(b * 255);
	// 		img[idx + 3] = 255;
	// 	}
	// 	// printf("\n");
	// }

	// for (int i = 0; i < bytes; i++)
	// {
	// 	printf("%4d: %4d\n", i + 1, img[i]);
	// 	if ((i + 1) % 4 == 0)
	// 		printf("---------\n");
	// }

	stbi_write_png("outImage/output.png", width, height, channels, img, width * channels);
	printf("Completed!\n");

	cudaCheckError(cudaFree(d_Img));
	free(img);
	std::cin.get();
}
