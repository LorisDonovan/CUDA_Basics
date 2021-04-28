# CUDA_Basics
This repo contains basic CUDA Programming examples.

# References
* CoffeeBeforeArch's [CUDA Crash Course](https://youtube.com/playlist?list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU)
* NVIDIA's [CUDA by Example](https://developer.download.nvidia.com/books/cuda-by-example/cuda-by-example-sample.pdf)

# Getting Started
For Windows 
* Run ```Win-GenerateProjects.bat``` 
* Set Project>Build Dependencies>Build Configuration to CUDA
* Copy ```glut64.dll``` from ```CUDA_Basics/dependencies/cudaByExample/bin/``` to ```bin/Release-windows-x86_64/CUDA_Basics/``` (for Release configuration) or ```bin/Debug-windows-x86_64/CUDA_Basics/``` (for Debug configuration)
* Then build and run specific files

# Output
Screenshots of some of the output.\
output of ```raytracing.cu```\
![](images/raytracing.png)

output of ```heatSim.cu```\
![](images/heatsim.png)

output of ```gpuJuliaSet.cu```\
![](images/juliaset.png)

output of ```ripples.cu```\
![](images/ripples.png)
