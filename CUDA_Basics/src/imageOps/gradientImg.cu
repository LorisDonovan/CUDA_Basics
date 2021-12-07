#include <cstdint>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "image.cuh"

__global__ void Kernel(uint8_t* outImg, const int32_t width, const int32_t height, const int32_t channels)
{
	int32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	int32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= width || y >= height)
		return;

	int32_t idx = (x + y * width) * channels;

	float r = float(x) / float(width);
	float g = float(y) / float(height);
	float b = 0.8f;
	
	outImg[idx + 0] = uint8_t(r * 255.99f);
	outImg[idx + 1] = uint8_t(g * 255.99f);
	outImg[idx + 2] = uint8_t(b * 255.99f);
	outImg[idx + 3] = 255;
}


int main(int argc, char** argv)
{
	constexpr float aspectRatio = 16.0f / 9.0f;
	constexpr uint32_t height   = 540;
	constexpr uint32_t width    = static_cast<uint32_t>(height * aspectRatio);
	constexpr uint32_t channels = 4; // number of channels in the image

	constexpr int32_t tx = 32;
	constexpr int32_t ty = 32;

	Image img(width, height, channels, "outImgage/output.png");
	
	dim3 threads(tx, ty);
	dim3 blocks((width + tx - 1) / tx, (height + ty - 1) / ty);
	Kernel<<<blocks, threads>>>(img.GetDevData(), width, height, channels);
	cudaCheckError(cudaGetLastError());

	img.WritePng();
	printf("Completed!\n");
}
