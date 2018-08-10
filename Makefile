
all: physics

cuda_path = /usr/local/cuda-9.2

libcudaphysics.so: cuda.cu cuda.h
	@nvcc -std=c++11 -arch=sm_50 --shared -g --compiler-options -fPIC -I$(cuda_path)/samples/common/inc cuda.cu -o libcudaphysics.so

physics: physics.cpp cpu.h gpu.h libcudaphysics.so helper.h collide_kernel.h grid.h
	@g++ -L$(cuda_path)/lib64 -std=c++11 -g -O3 -I. -I$(cuda_path)/include physics.cpp -o physics -fopenmp -lOpenCL libcudaphysics.so

debug: physics.cpp cpu.h gpu.h libcudaphysics.so helper.h collide_kernel.h grid.h
	@g++ -L. -L$(cuda_path)/lib64 -std=c++11 -g -O0 -I. -I$(cuda_path)/include physics.cpp -o debug -fopenmp -lOpenCL -lcudaphysics

clean:
	rm physics libcudaphysics.so debug > /dev/null
