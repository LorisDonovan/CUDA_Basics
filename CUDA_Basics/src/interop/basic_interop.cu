#include <iostream>

#include "cpu_bitmap.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#define cudaCheckErrors(val) CheckCuda(val, #val, __FILE__, __LINE__)

PFNGLBINDBUFFERARBPROC    glBindBuffer    = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers = NULL;
PFNGLGENBUFFERSARBPROC    glGenBuffers    = NULL;
PFNGLBUFFERDATAARBPROC    glBufferData    = NULL;

constexpr uint32_t width  = 512;
constexpr uint32_t height = 512;

GLuint bufferObj;
cudaGraphicsResource_t resource;

void CheckCuda(cudaError_t result, const char* func, const char* file, const uint32_t line);
static void KeyFunc(unsigned char key, int x, int y);
static void DrawFunc(void);

__global__ void Kernel(uchar4* ptr);

int main(int argc, char** argv)
{
	cudaDeviceProp  prop;
	int dev;

	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	cudaCheckErrors(cudaChooseDevice(&dev, &prop));
	// configure CUDA and OpenGL interop
	cudaCheckErrors(cudaGLSetGLDevice(dev));

	// initialize glut
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(width, height);
	glutCreateWindow("bitmap");

	glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
	glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
	glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
	glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");

	// create pixel buffer object
	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * 4, NULL, GL_DYNAMIC_DRAW_ARB);

	cudaCheckErrors(cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone));

	// do work with the memory dst being on the GPU, gotten via mapping
	cudaCheckErrors(cudaGraphicsMapResources(1, &resource, NULL));
	uchar4* devPtr;
	size_t  size;
	cudaCheckErrors(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource));

	dim3 grids(height / 16, width / 16);
	dim3 threads(16, 16);
	Kernel<<<grids, threads>>>(devPtr);
	cudaCheckErrors(cudaGraphicsUnmapResources(1, &resource, NULL));

	// set up GLUT and kick off main loop
	glutKeyboardFunc(KeyFunc);
	glutDisplayFunc(DrawFunc);
	glutMainLoop();

	return 0;
}

void CheckCuda(cudaError_t result, const char* func, const char* file, const uint32_t line)
{
	if (result)
	{
		std::cerr << "CUDA::ERROR::" << result << " in " << file << ": line " << line << " - '" << func << "'\n";
		cudaDeviceReset();
		__debugbreak();
	}
}

static void KeyFunc(unsigned char key, int x, int y) 
{
	switch (key) {
	case 27:
		// clean up OpenGL and CUDA
		cudaCheckErrors(cudaGraphicsUnregisterResource(resource));
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffers(1, &bufferObj);
		exit(0);
	}
}

static void DrawFunc(void) 
{
	// 0 as the last parameter => offset to the bitmap
	glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutSwapBuffers();
}


__global__ void Kernel(uchar4* ptr)
{
	int32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	int32_t y = threadIdx.y + blockDim.y * blockIdx.y;
	int32_t offset  = x + y * blockDim.x * gridDim.x;

	// calculate value at position
	float fx = x / (float)height - 0.5f;
	float fy = y / (float)width  - 0.5f;
	uint8_t green = 128 + 127 * sin(abs(fx * 100) - abs(fy * 100));

	ptr[offset].x = 0;
	ptr[offset].y = green;
	ptr[offset].z = 0;
	ptr[offset].w = 255;
}
