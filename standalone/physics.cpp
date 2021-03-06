#include "cpu.h"
#include "gpu.h"
#include "cuda.h"
#include "helper.h"

int main(int argc, char **argv) {
  if (argc == 4  || argc == 5) {
    int h = std::stoi(argv[1]);  // type of hardware (CPU = 0, GPU = 1)
    int num_threads = 64;         // number of threads (on CPU)
    int N = std::stoi(argv[2]);  // cells per dimension
    int cpc = 8;                 // collisions per cell
    REAL diameter = 40;          // diameter of the cell
    int T = std::stoi(argv[3]);  // number of iterations

    if (h == 2) {
      return cuda_collide(N, cpc, T, diameter, argc);
    }

    std::vector<int> indices(cpc*N*N*N);
    std::vector<std::array<REAL, 3>> positions;
    std::vector<std::array<REAL, 3>> force(N*N*N);
    std::vector<REAL> diameters(N * N * N, diameter);

    initialize(positions, N);

    omp_set_num_threads(num_threads);

    if (argc == 5) {
      // make random accessable pattern
      make_rai(&indices, N, cpc);
      std::cout << "=== Random Access Pattern ===" << std::endl;
    } else {
      // make regular accessable pattern
      make_rei(&indices, N, cpc);
      std::cout << "=== Regular Access Pattern ===" << std::endl;
    }

    auto t1 = Clock::now();
    if (h == 0) {
      std::cout << "Running on CPU with " << num_threads << " threads for " << T << " iterations" << std::endl << std::endl;
      cpu(positions, force, diameters, indices, N, T, cpc);
    } else if (h == 1) {
      std::cout << "Running on GPU (OpenCL) for " << T << " iterations" << std::endl << std::endl;
      opencl(positions, force, diameters, indices, N, T, cpc);
    } else {
      print_help();
    }
    auto t2 = Clock::now();
    std::cout << "\033[1mExecution time = "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()
              << " ms\033[0m" << std::endl;
    std::cout << "Total force = " << compute_sum(force) << std::endl;
    return 0;
  } else {
    print_help();
  }
}
