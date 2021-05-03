#pragma once

#include <string>


struct ShaderProgramSource
{
	std::string VertexSource;
	std::string FragmentSource;
};

class Shader
{
public:
	Shader(const std::string& filePath);
	~Shader();

	void Bind();
	void Unbind();

	void SetInt(const std::string& name, int value);

private:
	ShaderProgramSource Parser(const std::string& filePath);
	uint32_t CreateShader(const std::string& vertexSrc, const std::string& fragSrc);
	uint32_t CompileShader(uint32_t type, const std::string& source);

private:
	std::string m_FilePath;
	uint32_t m_RendererID;
};
