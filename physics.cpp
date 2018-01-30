#include "cpu.h"
#include "gpu.h"
#include "cuda.h"
#include "helper.h"
#include <fstream>

int main(int argc, char **argv) {
  if (argc == 6  || argc == 7) {
    int h = std::stoi(argv[1]);   // type of hardware (CPU = 0, GPU = 1)
    int num_threads = std::stoi(argv[5]);         // number of threads (on CPU)
    int N = std::stoi(argv[2]);   // cells per dimension
    int cpc = std::stoi(argv[4]); // collisions per cell
    REAL diameter = 40;           // diameter of the cell
    int T = std::stoi(argv[3]);   // number of iterations

    std::vector<int> indices(cpc*N*N*N);
    std::vector<std::array<REAL, 3>> positions;
    std::vector<std::array<REAL, 3>> force(N*N*N);
    std::vector<REAL> diameters(N * N * N, diameter);

    initialize(positions, N);

    omp_set_num_threads(num_threads);

    if (argc == 7) {
      // make random accessable pattern
      make_rai(&indices, N, cpc);
      // std::cout << "=== Random Access Pattern ===" << std::endl;
    } else {
      // make regular accessable pattern
      make_rei(&indices, N, cpc);
      // std::cout << "=== Regular Access Pattern ===" << std::endl;
    }

    float expected = calculate_expected(positions, &force, diameters, indices, N, T, cpc);

    if (h == 2) {
      return cuda_collide(N, cpc, T, diameter, argc, expected);
    }

    if (h == 0) {
      // std::cout << "Running on CPU with " << num_threads << " threads for " << T << " iterations" << std::endl << std::endl;
      for (int t = 0; t < T; t++) {
        auto t1 = Clock::now();
        cpu(positions, &force, diameters, indices, N, T, cpc);
        auto t2 = Clock::now();
        std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                     .count())
              << std::endl;
        FlushCache();
      }
      std::cout << std::endl;
    } else if (h == 1) {
      // std::cout << "Running on GPU (OpenCL) for " << T << " iterations" << std::endl << std::endl;
      opencl(positions, &force, diameters, indices, N, T, cpc, expected);
    } else {
      print_help();
    }
    // std::cout << "\033[1mExecution time = "
              // << " ms\033[0m" << std::endl;
    // std::cout << "Total force = " << compute_sum(force) << std::endl;
    
    // if (h == 0) {
    //   remove("cpu.txt");
    //   std::ofstream ofs("cpu.txt", std::ofstream::out);
    //   for (int k = 0; k < 0.5*force.size(); k++) {
    //     ofs << force[k][0] << ", " << force[k][1] << ", " << force[k][2] << std::endl;
    //   }
    //   for (int k = 0; k < 0.5*force.size(); k++) {
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
