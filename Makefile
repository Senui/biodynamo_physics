
all: physics

libcudaphysics.so: cuda.cu cuda.h
	@nvcc -std=c++11 -arch=sm_50 --shared --compiler-options -fPIC  cuda.cu -o libcudaphysics.so

physics: physics.cpp cpu.h gpu.h libcudaphysics.so helper.h
	@g++ -L/usr/local/cuda-8.0/lib64 -std=c++11 -O3 -I. -I/usr/local/cuda-8.0/include physics.cpp -o physics -fopenmp -lOpenCL libcudaphysics.so

clean:
	rm physics
