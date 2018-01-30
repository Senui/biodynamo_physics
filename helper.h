#include <array>
#include <iostream>
#include <random>
#include <vector>

#define REAL float

void print_help() {
  std::cout << "Usage: ./physics <H> <cells_per_dim> <iterations> <r>"
            << std::endl
            << std::endl << "<H> = hardware type; CPU = 0  |  OpenCL = 1  |  CUDA = 2"
            << std::endl << "<cells_per_dim> = cells per dimension"
            << std::endl << "<iterations> = number of iterations"
            << std::endl << "<neighbors> = number of neighbors"
            << std::endl << "<threads> = number of threads (CPU only)"
            << std::endl << "<r> = (optional) random access pattern [can be any value]"
            << std::endl;
}

void initialize(std::vector<std::array<REAL, 3>> &positions, int N) {
  positions.reserve(N * N * N);
  const REAL space = 10;
  for (size_t x = 0; x < N; x++) {
    REAL x_pos = x * space;
    for (size_t y = 0; y < N; y++) {
      REAL y_pos = y * space;
      for (size_t z = 0; z < N; z++) {
        positions.push_back({x_pos, y_pos, z * space});
      }
    }
  }
}

// make list of randomly ordered indices (rai)
void make_rai(std::vector<int>* rai, int N, int cpc) {
  std::random_device rd;
  std::mt19937 rng(4357);
  std::uniform_int_distribution<int> uni(0, N * N * N - 1);

  for (int i = 0; i < cpc * N * N * N; i++) {
    (*rai)[i] = uni(rng);
  }
}

// make list of self ordered indices (sei)
void make_sei(std::vector<int>* sei, int N, int cpc) {
  for (int i = 0; i < N * N * N; i++) {
    for (int j = 0; j < cpc; j++) {
      (*sei)[i * cpc + j] = i;
    }
  }
}

// make list of regular ordered indices (rei)
void make_rei(std::vector<int>* rei, int N, int cpc) {
  for (int i = 0; i < N * N * N; i++) {
    if (i < cpc / 2) {
      for (int k = 0; k < cpc; k++) {
        (*rei)[i * cpc + k] = k;
      }
    } 
    else if (i >= N*N*N - cpc / 2) {
      for (int k = 0; k < cpc; k++) {
        (*rei)[i * cpc + k] = N*N*N - cpc + k;
      }
    } else {
      int idx = 0;
      for (int j = -cpc / 2; j < cpc / 2 + 1; j++) {
        if (j != 0) {
          if (((i + j) >= 0) && ((i + j) < N*N*N)) {
            (*rei)[i * cpc + idx] = i + j;
          } else if (((i + j) == N*N*N)) {
            (*rei)[i * cpc + idx] = N*N*N - 1;
          } else {
            (*rei)[i * cpc + idx] = i * cpc + idx;
          }
          idx++;
        }
      }
    }
  }
}
