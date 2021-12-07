#pragma once

#include <cstdint>
#include <cuda_runtime.h>

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

class Image
{
public:
	Image(int32_t width, int32_t height, int32_t channels, const char* filename = "output.png")
		: m_Width(width), m_Height(height), m_Channels(channels), m_Filename(filename)
	{
		m_Size = width * height * channels;
		m_Data = new uint8_t[m_Size];

		cudaCheckError(cudaMalloc((void**)&m_DevData, m_Size * sizeof(uint8_t)));
		stbi_flip_vertically_on_write(true);
	}

	~Image() 
	{
		delete[] m_Data;
		cudaCheckError(cudaFree(m_DevData));
	}

	inline uint8_t* GetDevData() { return m_DevData; }

	void WritePng()
	{
		cudaCheckError(cudaMemcpy(m_Data, m_DevData, m_Size * sizeof(uint8_t), cudaMemcpyDeviceToHost));
		stbi_write_png(m_Filename, m_Width, m_Height, m_Channels, m_Data, m_Width * m_Channels);
	}

private:
	uint8_t* m_Data;
	uint8_t* m_DevData;

	size_t   m_Size;

	int32_t  m_Width;
	int32_t  m_Height;
	int32_t  m_Channels;

	const char* m_Filename;
};
