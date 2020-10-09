workspace "CUDA_Basics"

	architecture "x64"
	startproject "CUDA_Basics"

	configurations
	{
		"Debug",
		"Release"
	}

	flags
	{
		"MultiProcessorCompile"
	}

--Output directories
outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

project "CUDA_Basics"
	location "CUDA_Basics"
	kind "ConsoleApp"
	language "C++"
	cppdialect "C++17"
	staticruntime "off"

	targetdir("bin/" ..outputdir.. "/%{prj.name}")
	objdir("bin-int/" ..outputdir.. "/%{prj.name}")

	--files
	--{
	--	"%{prj.name}/src/**.cu",
	--	"%{prj.name}/src/**.cpp",
	--	"%{prj.name}/src/**.h"
	--}

	includedirs
	{
		"CUDA_Basics/src",
	}

	links
	{
		"cudart_static.lib",
		"cublas.lib",
		"curand.lib"
	}

	filter "system:windows"
		systemversion "latest"

		filter "configurations:Debug"
			runtime "Debug"
			symbols "on"

		filter "configurations:Debug"
			runtime "Debug"
			optimize "on"
