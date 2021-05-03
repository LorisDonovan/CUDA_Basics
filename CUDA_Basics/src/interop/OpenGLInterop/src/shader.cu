#include "shader.h"

#include <vector>
#include <sstream>
#include <fstream>
#include <cassert>
#include <iostream>

#include <glad/glad.h>


Shader::Shader(const std::string& filePath)
	:m_FilePath(filePath)
{
	ShaderProgramSource source = Parser(filePath);
	m_RendererID = CreateShader(source.VertexSource, source.FragmentSource);
}

Shader::~Shader()
{
	glDeleteProgram(m_RendererID);
}

void Shader::Bind()
{
	glUseProgram(m_RendererID);
}

void Shader::Unbind()
{
	glUseProgram(0);
}

void Shader::SetInt(const std::string& name, int value)
{
	GLint location = glGetUniformLocation(m_RendererID, name.c_str());
	glUniform1i(location, value);
}

ShaderProgramSource Shader::Parser(const std::string& filePath)
{
	enum class ShaderType
	{
		NONE = -1, VERTEX = 0, FRAGMENT = 1
	};

	std::ifstream stream(filePath);
	ShaderType type = ShaderType::NONE;
	std::string line;
	std::stringstream ss[2];

	while (getline(stream, line))
	{
		if (line.find("#shader") != std::string::npos)
		{
			if (line.find("vertex") != std::string::npos)
				type = ShaderType::VERTEX;
			else if (line.find("fragment") != std::string::npos)
				type = ShaderType::FRAGMENT;
		}
		else
		{
			ss[(int32_t)type] << line << '\n';
		}
	}

	stream.close();
	//std::cout << "Vertex Shader\n" << ss[0].str() << "\nFragment Shader\n" << ss[1].str() << std::endl;
	return { ss[0].str(), ss[1].str() };
}

uint32_t Shader::CreateShader(const std::string& vertexSrc, const std::string& fragSrc)
{
	uint32_t program = glCreateProgram();
	// compiling
	uint32_t vs = CompileShader(GL_VERTEX_SHADER, vertexSrc);
	uint32_t fs = CompileShader(GL_FRAGMENT_SHADER, fragSrc);

	// linking
	glAttachShader(program, vs);
	glAttachShader(program, fs);
	glLinkProgram(program);
	glValidateProgram(program);

	// delete the intermediates
	glDeleteShader(vs);
	glDeleteShader(fs);

	return program;
}

uint32_t Shader::CompileShader(uint32_t type, const std::string& source)
{
	uint32_t id = glCreateShader(type);
	const char* src = source.c_str();
	glShaderSource(id, 1, &src, nullptr);
	glCompileShader(id);

	// Error Handling
	int32_t result;
	glGetShaderiv(id, GL_COMPILE_STATUS, &result);
	if (result == GL_FALSE)
	{
		GLint maxLength = 0;
		glGetShaderiv(id, GL_INFO_LOG_LENGTH, &maxLength);

		std::vector<GLchar> infoLog(maxLength);
		glGetShaderInfoLog(id, maxLength, &maxLength, &infoLog[0]);

		glDeleteShader(id);

		if (type == GL_VERTEX_SHADER)
			std::cout << "Failed To Compile Vertex Shader!" << std::endl;
		if (type == GL_FRAGMENT_SHADER)
			std::cout << "Failed To Compile Fragment Shader!" << std::endl;
		std::cout << infoLog.data() << std::endl;
		assert(false);
	}

	return id;
}
