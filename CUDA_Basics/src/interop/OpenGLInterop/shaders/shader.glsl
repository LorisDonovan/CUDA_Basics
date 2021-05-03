#shader vertex
#version 450 core

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Color;

out vec3 Color;

void main()
{
	Color = a_Color;
	gl_Position = vec4(a_Position, 1.0f);
}

#shader fragment
#version 450 core

out vec4 FragColor;
in  vec3 Color;

void main()
{
	FragColor = vec4(Color, 1.0f);
}
