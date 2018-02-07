
all: physics

libcudaphysics.so: cuda.cu cuda.h
	@nvcc -std=c++11 -arch=sm_50 --shared -g --compiler-options -fPIC -I/usr/local/cuda-8.0/samples/common/inc cuda.cu -o libcudaphysics.so

physics: physics.cpp cpu.h gpu.h libcudaphysics.so helper.h collide_kernel.h grid.h
	@g++ -L/usr/local/cuda-8.0/lib64 -std=c++11 -g -O3 -I. -I/usr/local/cuda-8.0/include physics.cpp -o physics -fopenmp -lOpenCL libcudaphysics.so

debug: physics.cpp cpu.h gpu.h libcudaphysics.so helper.h collide_kernel.h grid.h
	@g++ -L/usr/local/cuda-8.0/lib64 -std=c++11 -g -O0 -I. -I/usr/local/cuda-8.0/include physics.cpp -o debug -fopenmp -lOpenCL libcudaphysics.so

clean:
	rm physics
