#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#define REAL float

#define STRINGIFY(src) #src

static const std::string source = STRINGIFY(__kernel void collide(
    __global float *positions, __global float *diameters,
    __global float *result, __global int *nidc, int N, int cpc) {
  size_t i = get_global_id(0);
  if (i < N * N * N) {
    for (int nb = 0; nb < cpc; nb++) {
      float r1 = 0.5 * diameters[i];
      float r2 = 0.5 * diameters[nidc[cpc * i + nb]];
      // We take virtual bigger radii to have a distant interaction, to get a
      // desired density.
      float additional_radius = 10.0 * 0.15;
      r1 += additional_radius;
      r2 += additional_radius;

      float comp1 =
          positions[3 * i + 0] - positions[3 * nidc[cpc * i + nb] + 0];
      float comp2 =
          positions[3 * i + 1] - positions[3 * nidc[cpc * i + nb] + 1];
      float comp3 =
          positions[3 * i + 2] - positions[3 * nidc[cpc * i + nb] + 2];
      float center_distance =
          sqrt(comp1 * comp1 + comp2 * comp2 + comp3 * comp3);

      // the overlap distance (how much one penetrates in the other)
      float delta = r1 + r2 - center_distance;

      if (delta < 0) {
        result[3 * i + 0] = 0;
        result[3 * i + 1] = 0;
        result[3 * i + 2] = 0;
        continue;
      }

      // to avoid a division by 0 if the centers are (almost) at the same
      // location
      if (center_distance < 0.00000001) {
        result[3 * i + 0] = 42;
        result[3 * i + 1] = 42;
        result[3 * i + 2] = 42;
        continue;
      }

      // the force itself
      float r = (r1 * r2) / (r1 + r2);
      float gamma = 1; // attraction coeff
      float k = 2;     // repulsion coeff
      float f = k * delta - gamma * sqrt(r * delta);

      float module = f / center_distance;
      result[3 * i + 0] = module * comp1;
      result[3 * i + 1] = module * comp2;
      result[3 * i + 2] = module * comp3;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
});

int opencl(std::vector<std::array<REAL, 3>>& positions,
           std::vector<std::array<REAL, 3>>* force,
           std::vector<REAL>& diameters, std::vector<int>& nidc, size_t N,
           int T, int cpc) {
  try {
    // Get list of OpenCL platforms.
    std::vector<cl::Platform> platform;
    cl::Platform::get(&platform);

    if (platform.empty()) {
      std::cerr << "OpenCL platforms not found." << std::endl;
    }

    // Get all available GPU devices
    cl::Context context;
    std::vector<cl::Device> devices;
    for (auto p = platform.begin(); p != platform.end(); p++) {
      std::vector<cl::Device> pldev;

      try {
        p->getDevices(CL_DEVICE_TYPE_ALL, &pldev);

        for (auto d = pldev.begin(); d != pldev.end(); d++) {
          if (!d->getInfo<CL_DEVICE_AVAILABLE>())
            continue;

          std::string ext = d->getInfo<CL_DEVICE_EXTENSIONS>();

          devices.push_back(*d);
        }
      } catch (...) {
        devices.clear();
        return 1;
      }
    }

    context = cl::Context(devices);

    if (devices.empty()) {
      std::cerr << "No GPU found!" << std::endl;
    }

    std::cout << "Executing on " << devices.size() << " device(s)" << std::endl;

    std::vector<cl::CommandQueue> command_queues;
    for (int i = 0; i < devices.size(); i++) {
      std::cout << "Device [" << i << "] = " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
      cl::CommandQueue queue(context, devices[i], CL_QUEUE_PROFILING_ENABLE);
      command_queues.push_back(queue);
    }

    std::cout << std::endl; 

    // Compile OpenCL program for found devices.
    cl::Program program(
        context,
        cl::Program::Sources(
            1, std::make_pair(source.c_str(), strlen(source.c_str()))));

    try {
      program.build(devices);
    } catch (const cl::Error &) {
      std::cerr << "OpenCL compilation error" << std::endl
                << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
                << std::endl;
      return 1;
    }

    cl::Kernel collide(program, "collide");

    // Allocate device buffers and transfer input data to device.
    cl::Buffer pos(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   positions.size() * 3 * sizeof(REAL),
                   positions.data()->data());

    cl::Buffer dia(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   diameters.size() * sizeof(REAL), diameters.data());

    cl::Buffer idc(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   nidc.size() * sizeof(cl_int), nidc.data());

    // Set kernel parameters.
    collide.setArg(0, pos);
    collide.setArg(1, dia);
    collide.setArg(3, idc);
    collide.setArg(4, static_cast<cl_int>(N));
    collide.setArg(5, static_cast<cl_int>(cpc));

    // Launch kernel on the compute device.
    size_t offset = 0.5*force->size();
    for (int i = 0; i < T; i++) {
      int j = 0;
      for (auto q : command_queues) {
        cl::Buffer res(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                   offset * 3 * sizeof(REAL), force->data()->data() + j*offset);
        collide.setArg(2, res);
        q.enqueueNDRangeKernel(collide, cl::NullRange, cl::NDRange(N * N * N), cl::NDRange(128));
        // Get result back to host.
        q.enqueueReadBuffer(res, CL_TRUE, 0, offset * 3 * sizeof(REAL), force->data() + j*offset);
        j++;
      }
    }
  } catch (const cl::Error &err) {
    std::cerr << "OpenCL error: " << err.what() << "(" << err.err() << ")"
              << std::endl;
    return 1;
  }
}
