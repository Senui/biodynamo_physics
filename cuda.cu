#include "cuda.h"

typedef std::chrono::high_resolution_clock Clock;

void initialize(float* positions, int N) {
  const float space = 10;
  int i = 0;
  for (size_t x = 0; x < N; x++) {
    float x_pos = x * space;
    for (size_t y = 0; y < N; y++) {
      float y_pos = y * space;
      for (size_t z = 0; z < N; z++) {
        positions[3*i+0] = x_pos;
        positions[3*i+1] = y_pos;
        positions[3*i+2] = z * space;
        i++;
      }
    }
  }
}

float compute_sum(float* voa, int N) {
  float sum = 0.0;

  for (int j = 0; j < N*N*N; j++) {
    sum += voa[3*j+0];
    sum += voa[3*j+1];
    sum += voa[3*j+2];
  }
  return sum;
}

// make list of randomly ordered indices (rai)
void make_rai(int* rai, int N, int cpc) {
  std::random_device rd;
  std::mt19937 rng(time(0));
  std::uniform_int_distribution<int> uni(0, N * N * N);

  for (int i = 0; i < cpc * N * N * N; i++) {
    rai[i] = uni(rng);
  }
}

// make list of regular ordered indices (rei)
void make_rei(int* rei, int N, int cpc) {
  for (int i = 0; i < N * N * N; i++) {
    if (i < cpc / 2) {
      for (int k = 0; k < cpc; k++) {
        rei[i * cpc + k] = k;
      }
    }
    else if (i >= N*N*N - cpc / 2) {
      for (int k = 0; k < cpc; k++) {
        rei[i * cpc + k] = N*N*N - cpc + k;
      }
    } else {
      int idx = 0;
      for (int j = -cpc / 2; j < cpc / 2 + 1; j++) {
        if (j != 0) {
          if (((i + j) >= 0) && ((i + j) < N*N*N)) {
            rei[i * cpc + idx] = i + j;
          } else if (((i + j) == N*N*N)) {
            rei[i * cpc + idx] = N*N*N - 1;
          } else {
            rei[i * cpc + idx] = i * cpc + idx;
          }
          idx++;
        }
      }
    }
  }
}

__global__ void collide(
       float* positions,
       float* diameters,
       float* result,
       int* nidc,
       int N,
       int cpc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
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
  }
}
 
int cuda_collide(int N, int cpc, int T, int diameter, int argc) {
  int* indices;
  float* positions;
  float* force;
  float* diameters;

  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&indices, cpc*N*N*N*sizeof(int));
  cudaMallocManaged(&positions, 3*N*N*N*sizeof(float));
  cudaMallocManaged(&force, 3*N*N*N*sizeof(float));
  cudaMallocManaged(&diameters, N*N*N*sizeof(float));
 
  // initialize
  const float space = 10;
  int i = 0;
  for (size_t x = 0; x < N; x++) {
    float x_pos = x * space;
    for (size_t y = 0; y < N; y++) {
      float y_pos = y * space;
      for (size_t z = 0; z < N; z++) {
        positions[3*i+0] = x_pos;
        positions[3*i+1] = y_pos;
        positions[3*i+2] = z * space;
        i++;
      }
    }
  }

  for (int j = 0; j < N*N*N; j++) {
    diameters[j] = diameter;
  }

  if (argc == 5) {
    // make random accessable pattern
    make_rai(indices, N, cpc);
    std::cout << "Running on GPU (CUDA) for " << T << " iterations (random access pattern)" << std::endl << std::endl;
  } else {
    // make regular accessable pattern
    make_rei(indices, N, cpc);
    std::cout << "Running on GPU (CUDA) for " << T << " iterations (regular access pattern)" << std::endl << std::endl;
  }


  auto t1 = Clock::now();
 
  for (int t = 0; t < T; t++) {
    // Launch kernel on 1M elements on the GPU
    int blockSize = 128;
    int numBlocks = (N*N*N + blockSize - 1) / blockSize;
    collide<<<numBlocks, blockSize>>>(positions, diameters, force, indices, N, cpc);
   
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
  }
 
  auto t2 = Clock::now();
  std::cout << "\033[1mExecution time = "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                   .count()
            << " ms\033[0m" << std::endl;
  std::cout << "Total force = " << compute_sum(force, N) << std::endl;
 
  // Free memory
  cudaFree(indices);
  cudaFree(positions);
  cudaFree(force);
  cudaFree(diameters);

  return 0;
}
