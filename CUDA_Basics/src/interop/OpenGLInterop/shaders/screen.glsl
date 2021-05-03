#shader vertex
#version 450 core

layout(location = 0) in vec2 a_Position;
layout(location = 1) in vec2 a_TexCoords;

out vec2 TexCoords;

void main()
{
	TexCoords   = a_TexCoords;
	gl_Position = vec4(a_Position, 0.0f, 1.0f);
}

#shader fragment
#version 450 core

out vec4 FragColor;
in  vec2 TexCoords;

uniform sampler2D u_ScreenTexture;

void main()
{
	vec3 col  = texture(u_ScreenTexture, TexCoords).rgb;
	FragColor = vec4(col, 1.0f);
}
