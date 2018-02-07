#include "cpu.h"
#include "gpu.h"
#include "grid.h"
#include "cuda.h"
#include "helper.h"
#include <fstream>

#include "libmorton/include/morton.h"

int main(int argc, char **argv) {
  if (argc == 5) {
    int h = std::stoi(argv[1]);   // type of hardware (CPU = 0, GPU = 1)
    int num_threads = std::stoi(argv[4]); // number of threads (on CPU)
    int N = std::stoi(argv[2]);   // cells per dimension
    REAL diameter = 30;           // diameter of the cell
    int T = std::stoi(argv[3]);   // number of iterations

    std::vector<std::array<REAL, 3>> positions;
    std::vector<std::array<REAL, 3>> force(N*N*N);
    std::vector<REAL> diameters(N * N * N, diameter);

    initialize(positions, N);
    // morton_sort(&positions);

    // Grid-specific operations
    Grid& g = Grid::GetInstance();
    g.Initialize(&positions, diameter);
    // g.PrintSuccessors();

    // OpenCL data
    std::vector<cl_uint> successors(positions.size());
    std::vector<cl_uint> gpu_starts;
    std::vector<cl_ushort> gpu_lengths;
    cl_uint box_length;
    std::array<cl_uint, 3> num_boxes_axis;
    std::array<cl_int, 3> grid_dimensions;

    // CUDA data
    std::vector<uint32_t> successors_cd(positions.size());
    std::vector<uint32_t> gpu_starts_cd;
    std::vector<uint16_t> gpu_lengths_cd;
    uint32_t box_length_cd;
    std::array<uint32_t, 3> num_boxes_axis_cd;
    std::array<int32_t, 3> grid_dimensions_cd;

    g.GetSuccessors(&successors);
    g.GetGPUBoxData(&gpu_starts, &gpu_lengths);
    g.GetGridData(&box_length, num_boxes_axis, grid_dimensions);

    g.GetSuccessors(&successors_cd);
    g.GetGPUBoxData(&gpu_starts_cd, &gpu_lengths_cd);
    g.GetGridData(&box_length_cd, num_boxes_axis_cd, grid_dimensions_cd);

    omp_set_num_threads(num_threads);

    float expected = 0.0;
    if (h == 1 || h == 2) {
        expected = calculate_expected(positions, &force, diameters, g, N);
        clear_force_cpu(&force);
    }


    if (h == 2) {
      return cuda_collide(&gpu_starts_cd, &gpu_lengths_cd, &successors_cd, box_length_cd, &num_boxes_axis_cd, &grid_dimensions_cd,  N, T, diameter, expected);
    }

    if (h == 0) {
      // std::cout << "Running on CPU with " << num_threads << " threads for " << T << " iterations" << std::endl << std::endl;
      for (int t = 0; t < T; t++) {
        auto t1 = Clock::now();
        cpu(positions, &force, diameters, g, N);
        auto t2 = Clock::now();
        std::cout << (std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()) << " ms"
                  << std::endl;
        FlushCache();
      }
    } else if (h == 1) {
      // std::cout << "Running on GPU (OpenCL) for " << T << " iterations" << std::endl << std::endl;
      opencl(positions, diameters, &force, &gpu_starts, &gpu_lengths, &successors, box_length, &num_boxes_axis, &grid_dimensions, N, T, expected);
    } else {
      print_help();
    }

    // std::cout << "\033[1mExecution time = "
    //           << " ms\033[0m" << std::endl;
    // std::cout << "Total force = " << compute_sum(force) << std::endl;
    
    // if (h == 0) {
    //   remove("cpu.txt");
    //   std::ofstream ofs("cpu.txt", std::ofstream::out);
    //   for (int k = 0; k < force.size(); k++) {
    //     ofs << force[k][0] << ", " << force[k][1] << ", " << force[k][2] << std::endl;
    //   }
    //   ofs.close();
    // } else if (h == 1) {
    //   remove("gpu.txt");
    //   std::ofstream ofs("gpu.txt", std::ofstream::out);
    //   for (int k = 0; k < force.size(); k++) {
    //     ofs << force[k][0] << ", " << force[k][1] << ", " << force[k][2] << std::endl;
    //   }
    //   ofs.close();
    // }
    return 0;
  } else {
    print_help();
  }
}
