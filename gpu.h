#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "grid.h"
#include "collide_kernel.h"

REAL compute_sum_ocl(std::vector<std::array<REAL, 3>>& voa) {
  REAL sum = 0.0;

  for (auto& arr : voa) {
    sum += fabs(arr[0]);
    sum += fabs(arr[1]);
    sum += fabs(arr[2]);
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

bool are_same(REAL a, REAL b) {
  return fabs(a - b) < 1e-20;
}

int opencl(std::vector<std::array<REAL, 3>>& positions, 
                  std::vector<REAL>& diameters,
                  std::vector<std::array<REAL, 3>>* force,
                  std::vector<cl_uint>* starts,
                  std::vector<cl_ushort>* lengths,
                  std::vector<cl_uint>* successors,
                  cl_uint box_length,
                  std::array<cl_uint, 3>* num_boxes_axis,
                  std::array<cl_int, 3>* grid_dimensions,
                  size_t N, int T, REAL expected) {
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

    std::cout << device[0].getInfo<CL_DEVICE_NAME>() << std::endl;

    // Create command queue.
    cl::CommandQueue queue(context, device[0], CL_QUEUE_PROFILING_ENABLE);

    // Compile OpenCL program for found device.
    cl::Program program(
        context,
        cl::Program::Sources(1, std::make_pair(collide_kernel, strlen(collide_kernel))));

    std::cout << "Compiling OpenCL kernel" << std::endl;
    try {
      program.build(device);
    } catch (const cl::Error &) {
      std::cerr << "OpenCL compilation error" << std::endl
                << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device[0])
                << std::endl;
      return 1;
    }

    cl::Kernel collide(program, "collide");
    cl::Kernel clear_force_opencl(program, "clear_force_opencl");
    std::cout << "Creating OpenCL buffers" << std::endl;

    // Allocate device buffers and transfer input data to device.
    cl::Buffer positions_arg(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 positions.size() * 3 * sizeof(cl_float), positions.data()->data());

    cl::Buffer diameters_arg(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 diameters.size() * sizeof(cl_float), diameters.data());

    cl::Buffer force_arg(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                 force->size() * 3 * sizeof(cl_float), force->data()->data());

    cl::Buffer starts_arg(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 starts->size() * sizeof(cl_uint), starts->data());

    cl::Buffer lengths_arg(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 lengths->size() * sizeof(cl_short), lengths->data());

    cl::Buffer successors_arg(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 successors->size() * sizeof(cl_uint), successors->data());

    cl::Buffer nba_arg(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 3 * sizeof(cl_uint), num_boxes_axis->data());

    cl::Buffer gd_arg(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 3 * sizeof(cl_int), grid_dimensions->data());

    // Set kernel parameters.
    cl_int err0 = collide.setArg(0, positions_arg);
    cl_int err1 = collide.setArg(1, diameters_arg);
    cl_int err2 = collide.setArg(2, static_cast<cl_int>(N));
    cl_int err3 = collide.setArg(3, starts_arg);
    cl_int err4 = collide.setArg(4, lengths_arg);
    cl_int err5 = collide.setArg(5, successors_arg);
    cl_int err6 = collide.setArg(6, box_length);
    cl_int err7 = collide.setArg(7, nba_arg);
    cl_int err8 = collide.setArg(8, gd_arg);
    cl_int err9 = collide.setArg(9, force_arg);

    clear_force_opencl.setArg(0, force_arg);
    clear_force_opencl.setArg(1, static_cast<cl_int>(N));

    int block_size = 128;

    std::cout << "Launching OpenCL kernels" << std::endl;
    // Launch kernel on the compute device.
    for (int i = 0; i < T; i ++) {
      // std::cout << "global work size = " << (N*N*N + (block_size - (N*N*N)%block_size)) << std::endl;
      auto t1 = Clock::now();
      queue.enqueueNDRangeKernel(collide, cl::NullRange, cl::NDRange(N*N*N + (block_size - (N*N*N)%block_size)), cl::NDRange(block_size));
      queue.enqueueReadBuffer(force_arg, CL_TRUE, 0, force->size() * 3 * sizeof(REAL), force->data()->data());
      auto t2 = Clock::now();
      std::cout << (std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()) << " ms"
                << std::endl;

      REAL actual = compute_sum_ocl(*force);
      // std::cout << "Iteration " << i << ": " << actual << std::endl;
      if (are_same(actual, expected)) {
        // std::cout << "Correct result! Because " << std::setprecision(15) << actual << " == " << expected << std::endl;
        clear_force(force);
        queue.enqueueNDRangeKernel(clear_force_opencl, cl::NullRange, cl::NDRange(N*N*N + (block_size - (N*N*N)%block_size)), cl::NDRange(block_size));
        continue;    
      } else {
        std::cout << "Wrong result! Difference = " << fabs(actual - expected) << std::endl;
        std::cout << "Should be " << expected << std::endl;
        return 1;
      }
    }
  } catch (const cl::Error &err) {
    std::cerr << "OpenCL error: " << err.what() << "(" << err.err() << ")"
              << std::endl;
    return 1;
  }
}