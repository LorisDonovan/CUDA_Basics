#include <cmath>
#include <ctime>
#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#include "shader.h"

#define cudaCheckErrors(val) CheckCuda(val, #val, __FILE__, __LINE__)


// window properties
const uint32_t width  = 960;
const uint32_t height = 540;

// interop resources
cudaResourceDesc       resourceDesc;
cudaGraphicsResource_t textureResource;
cudaArray_t            textureArray;
cudaSurfaceObject_t    surfaceObj = 0;


// OpenGL
GLFWwindow* InitWindow();
void InitFbQuad(Shader& screen, uint32_t& quadVao, uint32_t& quadVbo);
uint32_t InitGLTexture();
void Cleanup(uint32_t& quadVao, uint32_t& quadVbo);

// glfw
void ProcessInput(GLFWwindow* window);
void FramebufferSizeCallback(GLFWwindow* window, int32_t width, int32_t height);

// CUDA
void CheckCuda(cudaError_t result, const char* func, const char* file, const uint32_t line);
int32_t InitCudaDevice();
void InitCudaTexture(uint32_t textureID);
__global__ void Kernel(cudaSurfaceObject_t surfaceObj, int32_t tick);


int main(int argc, char** argv)
{
	// initialize CUDA for OpenGL interop and GLFW window
	int32_t deviceID    = InitCudaDevice();
	GLFWwindow* window  = InitWindow();

	// initialize quad for screen framebuffer
	Shader screen("src/interop/OpenGLInterop/shaders/screen.glsl");
	uint32_t quadVao, quadVbo;
	InitFbQuad(screen, quadVao, quadVbo);

	// initialize GL texture for CUDA surface object
	uint32_t textureID  = InitGLTexture();
	InitCudaTexture(textureID);

	// CUDA kernel thread layout
	int32_t numThreads = 32;
	dim3 blocks((width + numThreads - 1) / numThreads, (height + numThreads - 1) / numThreads);
	dim3 threads(numThreads, numThreads);

	double  lastTime = glfwGetTime();
	int32_t nbFrames = 0;

	int32_t ticks = 1;
	while (!glfwWindowShouldClose(window))
	{
		// Measure speed
		double currentTime = glfwGetTime();
		nbFrames++;
		if (currentTime - lastTime >= 1.0) // If last prinf() was more than 1 sec ago
		{ 
			// print and reset timer
			printf("\r%10.5f ms/frame", 1000.0 / double(nbFrames));
			nbFrames = 0;
			lastTime += 1.0;
		}

		ProcessInput(window);

		// CUDA register and create surface object resource
		cudaCheckErrors(cudaGraphicsMapResources(1, &textureResource));
		cudaCheckErrors(cudaGraphicsSubResourceGetMappedArray(&textureArray, textureResource, 0, 0));
		resourceDesc.res.array.array = textureArray;
		cudaCheckErrors(cudaCreateSurfaceObject(&surfaceObj, &resourceDesc));
		Kernel<<<blocks, threads>>>(surfaceObj, ticks++);
		cudaCheckErrors(cudaGraphicsUnmapResources(1, &textureResource)); // sync cuda operations before graphics calls

		// Render
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		screen.Bind();
		glBindVertexArray(quadVao);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, textureID);
		glDrawArrays(GL_TRIANGLES, 0, 6);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	
	Cleanup(quadVao, quadVbo);
	glfwTerminate();
	return 0;
}


// OpenGL------------------------------------------------------------
GLFWwindow* InitWindow()
{
	// initialize and configure glfw
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// create a windowed mode window and its OpenGL context
	GLFWwindow* window = glfwCreateWindow(width, height, "CUDA-OpenGL-Interop", NULL, NULL);
	if (!window)
	{
		std::cout << "ERROR: Failed to create GLFW Window" << std::endl;
		glfwTerminate();
		exit(-1);
	}

	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "ERROR: Failed to initialize GLAD" << std::endl;
		exit(-1);
	}

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glDisable(GL_DEPTH_TEST);
	glViewport(0, 0, width, height);
	glfwSetFramebufferSizeCallback(window, FramebufferSizeCallback);

	std::cout << "OpenGL Info:\n";
	std::cout << "    Vendor:   " << glGetString(GL_VENDOR)   << "\n";
	std::cout << "    Renderer: " << glGetString(GL_RENDERER) << "\n";
	std::cout << "    Version:  " << glGetString(GL_VERSION)  << std::endl;

	return window;
}

void InitFbQuad(Shader& screen, uint32_t& quadVao, uint32_t& quadVbo)
{
	float quadVertices[] = {
		// positions   // texCoords
		-1.0f,  1.0f,  0.0f, 1.0f,
		-1.0f, -1.0f,  0.0f, 0.0f,
		 1.0f, -1.0f,  1.0f, 0.0f,

		-1.0f,  1.0f,  0.0f, 1.0f,
		 1.0f, -1.0f,  1.0f, 0.0f,
		 1.0f,  1.0f,  1.0f, 1.0f
	};
	
	glGenBuffers(1, &quadVbo);
	glGenVertexArrays(1, &quadVao);

	glBindVertexArray(quadVao);
	glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
	// position attribute
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// texCoords attribute
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	glEnableVertexAttribArray(1);

	screen.Bind();
	screen.SetInt("u_ScreenTexture", 0);
	screen.Unbind();
}

uint32_t InitGLTexture()
{
	uint32_t textureID;
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	// texture properties
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	// set texture image
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	// unbind texture
	glBindTexture(GL_TEXTURE_2D, 0);

	return textureID;
}

void Cleanup(uint32_t& quadVao, uint32_t& quadVbo)
{
	glDeleteBuffers(1, &quadVbo);
	glDeleteVertexArrays(1, &quadVao);
}


// glfw--------------------------------------------------------------
void ProcessInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

void FramebufferSizeCallback(GLFWwindow* window, int32_t width, int32_t height)
{
	glViewport(0, 0, width, height);
}


// CUDA--------------------------------------------------------------
void CheckCuda(cudaError_t result, const char* func, const char* file, const uint32_t line)
{
	if (result)
	{
		std::cout << "CUDA::ERROR::" << static_cast<uint32_t>(result) << " in " << file << " : line " << line << " - '" << func << "'" << std::endl;
		cudaDeviceReset();
		__debugbreak();
	}
}

int32_t InitCudaDevice()
{
	cudaDeviceProp prop;
	int32_t dev;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	// for surface object, compute capability >= 3.0
	prop.major = 3;
	prop.minor = 0;
	cudaCheckErrors(cudaChooseDevice(&dev, &prop)); // choose a cuda compatible device with compute capability 3.0 or better
	cudaCheckErrors(cudaGLSetGLDevice(dev));

	return dev;
}

void InitCudaTexture(uint32_t textureID)
{
	// register texture with CUDA resource
	cudaCheckErrors(cudaGraphicsGLRegisterImage(
		&textureResource,                          // resource
		textureID,                                 // image
		GL_TEXTURE_2D,                             // target
		cudaGraphicsRegisterFlagsSurfaceLoadStore  // flags
	));
	// resource description for surface
	memset(&resourceDesc, 0, sizeof(resourceDesc));
	resourceDesc.resType = cudaResourceTypeArray;
}

__global__ void Kernel(cudaSurfaceObject_t surfaceObj, int32_t tick)
{
	int32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	int32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < width && y < height)
	{
		float fx = x - width  * 0.5f;
		float fy = y - height * 0.5f;
		float d  = std::sqrtf(fx * fx + fy * fy);
		uint8_t grey = (uint8_t)(128.0f + 127.0f * std::cos(d / 10.0f - tick / 7.0f) / (d / 10.0f + 1.0f));

		uchar4 data = make_uchar4(grey, grey, grey, 255);
		surf2Dwrite(data, surfaceObj, x * sizeof(uchar4), y);
	}
}
