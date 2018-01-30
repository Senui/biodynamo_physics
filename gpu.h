#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#define REAL float

#define STRINGIFY(src) #src

static const std::string source = STRINGIFY(
__kernel void collide(
       __global float* positions,
       __global float* diameters,
       __global float* result,
       __global int* nidc,
       int N,
       int cpc) {
  size_t i = get_global_id(0);
  if (i < N*N*N) {
    for (int nb = 0; nb < cpc; nb++) {
      float r1 = 0.5 * diameters[i];
      float r2 = 0.5 * diameters[nidc[cpc * i + nb]];
      // We take virtual bigger radii to have a distant interaction, to get a
      // desired density.
      float additional_radius = 10.0 * 0.15;
      r1 += additional_radius;
      r2 += additional_radius;

      float comp1 = positions[3*i + 0] - positions[3*nidc[cpc * i + nb]+0];
      float comp2 = positions[3*i + 1] - positions[3*nidc[cpc * i + nb]+1];
      float comp3 = positions[3*i + 2] - positions[3*nidc[cpc * i + nb]+2];
      float center_distance = sqrt(comp1 * comp1 + comp2 * comp2 + comp3 * comp3);

      // the overlap distance (how much one penetrates in the other)
      float delta = r1 + r2 - center_distance;

      if (delta < 0) {
        result[3*i + 0] = 0;
        result[3*i + 1] = 0;
        result[3*i + 2] = 0;
        continue;
      }

      // to avoid a division by 0 if the centers are (almost) at the same
      //  location
      if (center_distance < 0.00000001) {
        result[3*i + 0] = 42;
        result[3*i + 1] = 42;
        result[3*i + 2] = 42;
        continue;
      }

      // the force itself
      float r = (r1 * r2) / (r1 + r2);
      float gamma = 1; // attraction coeff
      float k = 2;     // repulsion coeff
      float f = k * delta - gamma * sqrt(r * delta);

      float module = f / center_distance;
      result[3*i + 0] = module * comp1;
      result[3*i + 1] = module * comp2;
      result[3*i + 2] = module * comp3;
    }
      barrier(CLK_GLOBAL_MEM_FENCE);
  }
});

float compute_sum_ocl(std::vector<std::array<REAL, 3>>& voa) {
  float sum = 0.0;

  for (auto arr : voa) {
    for (int i = 0; i < 3; i++) {
      sum += arr[i];
    }
  }
  return sum;
}

void clear_force(std::vector<std::array<REAL, 3>>* voa) {
  for (int i = 0; i < voa->size(); i++) {
    (*voa)[i][0] = 0;
    (*voa)[i][1] = 0;
    (*voa)[i][2] = 0;
  }
}

bool are_same(float a, float b) {
  return fabs(a - b) < 0.0001;
}

int opencl(std::vector<std::array<REAL, 3>>& positions, 
                  std::vector<std::array<REAL, 3>>* force,
                  std::vector<REAL>& diameters,
                  std::vector<int>& nidc,
                  size_t N, int T, int cpc, float expected) {
  try {
    // Get list of OpenCL platforms.
    std::vector<cl::Platform> platform;
    cl::Platform::get(&platform);

    if (platform.empty()) {
      std::cerr << "OpenCL platforms not found." << std::endl;
    }

    // Get first available GPU device
    cl::Context context;
    std::vector<cl::Device> device;
    for (auto p = platform.begin(); device.empty() && p != platform.end();
         p++) {
      std::vector<cl::Device> pldev;

      try {
        p->getDevices(CL_DEVICE_TYPE_GPU, &pldev);

        for (auto d = pldev.begin(); device.empty() && d != pldev.end(); d++) {
          if (!d->getInfo<CL_DEVICE_AVAILABLE>())
            continue;

          std::string ext = d->getInfo<CL_DEVICE_EXTENSIONS>();

          device.push_back(*d);
          context = cl::Context(device);
        }
      } catch (...) {
        // std::cout << "Found bad platform... Continuing to next one." << std::endl;
        device.clear();
        // return 1;
        continue;
      }
    }

    if (device.empty()) {
      std::cerr << "No GPU found!" << std::endl;
    }

    // std::cout << device[0].getInfo<CL_DEVICE_NAME>() << std::endl;

    // Create command queue.
    cl::CommandQueue queue(context, device[0], CL_QUEUE_PROFILING_ENABLE);

    // Compile OpenCL program for found device.
    cl::Program program(
        context,
        cl::Program::Sources(1, std::make_pair(source.c_str(), strlen(source.c_str()))));

    try {
      program.build(device);
    } catch (const cl::Error &) {
      std::cerr << "OpenCL compilation error" << std::endl
                << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device[0])
                << std::endl;
      return 1;
    }

    cl::Kernel collide(program, "collide");

    // Allocate device buffers and transfer input data to device.
    cl::Buffer pos(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 positions.size() * 3 * sizeof(REAL), positions.data()->data());

    cl::Buffer dia(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 diameters.size() * sizeof(REAL), diameters.data());

    cl::Buffer res(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                 force->size() * 3 * sizeof(REAL), force->data()->data());

    cl::Buffer idc(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 nidc.size() * sizeof(cl_int), nidc.data());

    // Set kernel parameters.
    collide.setArg(0, pos);
    collide.setArg(1, dia);
    collide.setArg(2, res);
    collide.setArg(3, idc);
    collide.setArg(4, static_cast<cl_int>(N));
    collide.setArg(5, static_cast<cl_int>(cpc));

    int block_size = 1024;

    // Launch kernel on the compute device.
    for (int i = 0; i < T; i ++) {
      auto t1 = Clock::now();
      queue.enqueueNDRangeKernel(collide, cl::NullRange, cl::NDRange(N*N*N + (block_size - (N*N*N)%block_size)), cl::NDRange(block_size));
      queue.enqueueReadBuffer(res, CL_TRUE, 0, force->size() * 3 * sizeof(float), force->data());
      auto t2 = Clock::now();
      std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                     .count())
                << std::endl;
      float actual = compute_sum_ocl(*force);
      if (are_same(actual, expected)) {
        clear_force(force);
        continue;    
      } else {
        std::cout << "Wrong result! Difference = " << fabs(actual - expected) << std::endl;
        return 1;
      }
    }
  } catch (const cl::Error &err) {
    std::cerr << "OpenCL error: " << err.what() << "(" << err.err() << ")"
              << std::endl;
    return 1;
  }
}
