
all: physics

physics: physics.cpp cpu.h gpu.h
	@g++ -L/usr/local/cuda-8.0/lib64 -std=c++11 -g -O3 -I. -I/usr/local/cuda-8.0/include physics.cpp -o physics -fopenmp -lOpenCL libcuda.so

clean:
	rm physics
