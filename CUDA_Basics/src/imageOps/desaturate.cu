#include <cstdint>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

#define cudaCheckError(val) CheckCuda(val, #val, __FILE__, __LINE__)

inline void CheckCuda(cudaError_t result, const char* func, const char* file, const int32_t line)
{
	if (result)
	{
		printf("CUDA::ERROR_CODE::%d in function:'%s' file: '%s' line: %d\nERROR_DESCRIPTION:: %s\n", 
			result, func, file, line, cudaGetErrorString(result));
		exit(-1);
	}
}

__global__ void Kernel(const uint8_t* inImg, uint8_t* outImg, int32_t width, int32_t height, int32_t channels)
{
	int32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= width || y >= height)
		return;

	int32_t idx = (x + y * width) * channels;

	outImg[idx + 0] = uint8_t(0.33f * inImg[idx + 0]);
	outImg[idx + 1] = uint8_t(0.33f * inImg[idx + 1]);
	outImg[idx + 2] = uint8_t(0.33f * inImg[idx + 2]);
	outImg[idx + 3] = inImg[idx + 3];
}


int main()
{
	constexpr uint32_t tx = 16;
	constexpr uint32_t ty = 16;

	int32_t  width, height, channels;
	uint8_t* inImg, * d_InImg, * outImg, * d_OutImg;

	stbi_set_flip_vertically_on_load(true);
	inImg = stbi_load("outImage/input.png", &width, &height, &channels, 0);
	const int32_t imgSize = width * height * channels;
	printf("Image info:\n\twidth = %d, height = %d, channels = %d\n",
		width, height, channels);

	cudaCheckError(cudaMalloc((void**)&d_InImg,  imgSize * sizeof(uint8_t)));
	cudaCheckError(cudaMalloc((void**)&d_OutImg, imgSize * sizeof(uint8_t)));
	outImg = (uint8_t*)malloc(imgSize * sizeof(uint8_t));
	
	cudaCheckError(cudaMemcpy(d_InImg, inImg, imgSize * sizeof(uint8_t), cudaMemcpyHostToDevice));

	dim3 threads(tx, ty);
	dim3 blocks((tx + width - 1) / tx, (ty + height - 1) / ty);
	printf("Kernel info:\n\tthreads = (%d, %d), blocks = (%d, %d)\n", 
		threads.x, threads.y, blocks.x, blocks.y);
	Kernel<<<blocks, threads>>>(d_InImg, d_OutImg, width, height, channels);
	cudaCheckError(cudaGetLastError());

	cudaCheckError(cudaMemcpy(outImg, d_OutImg, imgSize * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	// for (int i = 0; i < imgSize; i++)
	// 	printf("%3d\n", img[i]);

	stbi_flip_vertically_on_write(true);
	stbi_write_png("outImage/output.png", width, height, channels, outImg, width * channels);
	printf("Completed!\n");

	cudaCheckError(cudaFree(d_InImg));
	cudaCheckError(cudaFree(d_OutImg));
	free(inImg);
	free(outImg);
}
