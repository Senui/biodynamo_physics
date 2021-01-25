#include <array>
#include "helper_math.h"
#include "cuda.h"

#include <fstream>

typedef std::chrono::high_resolution_clock Clock;

void initialize(REAL* positions, int N) {
  const REAL space = 10;
  int i = 0;
  for (size_t x = 0; x < N; x++) {
    REAL x_pos = x * space;
    for (size_t y = 0; y < N; y++) {
      REAL y_pos = y * space;
      for (size_t z = 0; z < N; z++) {
        positions[3*i+0] = x_pos;
        positions[3*i+1] = y_pos;
        positions[3*i+2] = z * space;
        i++;
      }
    }
  }
}

REAL compute_sum_cuda(REAL* voa, int N) {
  REAL sum = 0.0;

  for (int j = 0; j < N*N*N; j++) {
    sum += fabs(voa[3*j+0]);
    sum += fabs(voa[3*j+1]);
    sum += fabs(voa[3*j+2]);
  }
  return sum;
}

void clear_force(REAL* voa, int N) {
  for (int j = 0; j < N*N*N; j++) {
    voa[3*j+0] = 0;
    voa[3*j+1] = 0;
    voa[3*j+2] = 0;
  }
}

bool are_same(REAL a, REAL b) {
  return fabs(a - b) < std::numeric_limits<REAL>::epsilon();
}

__device__ int3 get_box_coordinates(REAL3 pos, int32_t* grid_dimensions, uint32_t box_length) {
  int3 box_coords;
  box_coords.x = (floor(pos.x) - grid_dimensions[0]) / box_length;
  box_coords.y = (floor(pos.y) - grid_dimensions[1]) / box_length;
  box_coords.z = (floor(pos.z) - grid_dimensions[2]) / box_length;
  return box_coords;
}

__device__ uint32_t get_box_id_2(int3 bc, uint32_t* num_boxes_axis) {
  return bc.z * num_boxes_axis[0]*num_boxes_axis[1] + bc.y * num_boxes_axis[0] + bc.x;
}

__device__ uint32_t get_box_id(REAL3 pos, uint32_t* num_boxes_axis, int32_t* grid_dimensions, uint32_t box_length) {
  int3 box_coords = get_box_coordinates(pos, grid_dimensions, box_length);
  return get_box_id_2(box_coords, num_boxes_axis);
}

__device__ void compute_force(REAL* positions, REAL* diameters, uint32_t idx, uint32_t nidx, REAL* result) {
  REAL r1 = 0.5 * diameters[idx];
  REAL r2 = 0.5 * diameters[nidx];
  // We take virtual bigger radii to have a distant interaction, to get a desired density.
  REAL additional_radius = 10.0 * 0.15;
  r1 += additional_radius;
  r2 += additional_radius;

  REAL comp1 = positions[3*idx + 0] - positions[3*nidx + 0];
  REAL comp2 = positions[3*idx + 1] - positions[3*nidx + 1];
  REAL comp3 = positions[3*idx + 2] - positions[3*nidx + 2];
  REAL center_distance = sqrtf(comp1 * comp1 + comp2 * comp2 + comp3 * comp3);

  // the overlap distance (how much one penetrates in the other)
  REAL delta = r1 + r2 - center_distance;

  if (delta < 0) {
    return;
  }

  // to avoid a division by 0 if the centers are (almost) at the same location
  if (center_distance < 0.00000001) {
    result[3*idx + 0] += 42.0;
    result[3*idx + 1] += 42.0;
    result[3*idx + 2] += 42.0;
    return;
  }

  // printf("Colliding cell [%d] and [%d]\n", idx, nidx);
  // printf("Delta for neighbor [%d] = %f\n", nidx, delta);

  // the force itself
  REAL r = (r1 * r2) / (r1 + r2);
  REAL gamma = 1; // attraction coeff
  REAL k = 2;     // repulsion coeff
  REAL f = k * delta - gamma * sqrt(r * delta);

  REAL module = f / center_distance;
  result[3*idx + 0] += module * comp1;
  result[3*idx + 1] += module * comp2;
  result[3*idx + 2] += module * comp3;
}


__device__ void default_force(REAL* positions,
                   REAL* diameters,
                   uint32_t idx, uint32_t start, uint16_t length,
                   uint32_t* successors,
                   REAL* result) {
  // printf("start = %d \n", start);
  // printf("length = %d \n", length);
  uint32_t nidx = start;
  for (uint16_t nb = 0; nb < length; nb++) {
    // implement logic for within radius here
    if (nidx != idx) {
      compute_force(positions, diameters, idx, nidx, result);
    }
    // traverse linked-list
    nidx = successors[nidx];
  }
  // printf("\n");
}

__global__ void collide(
       REAL* positions,
       REAL* diameters,
       int N,
       uint32_t* starts,
       uint16_t* lengths,
       uint32_t* successors,
       uint32_t* box_length,
       uint32_t* num_boxes_axis,
       int32_t* grid_dimensions,
       REAL* result) {
  uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tidx < N * N * N) {
    // if (tidx == 0) {
      REAL3 pos;
      pos.x = positions[3*tidx + 0];
      pos.y = positions[3*tidx + 1];
      pos.z = positions[3*tidx + 2];

      int3 box_coords = get_box_coordinates(pos, grid_dimensions, box_length[0]);

      // Moore neighborhood
      for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
          for (int x = -1; x <= 1; x++) {
            uint32_t bidx = get_box_id_2(box_coords + make_int3(x, y, z), num_boxes_axis);
            if (lengths[bidx] != 0) {
              // printf("Box %d\n", bidx);
              // printf("length = %d\n", lengths[bidx]);
              default_force(positions, diameters, tidx, starts[bidx], lengths[bidx], successors, result);
            }
          }
        }
      }
    }
  // }
}

int cuda_collide(std::vector<uint32_t>* starts,
                 std::vector<uint16_t>* lengths,
                 std::vector<uint32_t>* successors,
                 uint32_t box_length,
                 std::array<uint32_t, 3>* num_boxes_axis,
                 std::array<int32_t, 3>* grid_dimensions,
                 int N, int T, int diameter, REAL expected) {
  REAL* positions;
  REAL* force;
  REAL* diameters;

  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&positions, 3*N*N*N*sizeof(REAL));
  cudaMallocManaged(&force, 3*N*N*N*sizeof(REAL));
  cudaMallocManaged(&diameters, N*N*N*sizeof(REAL));

  uint32_t* d_starts = NULL;
  uint16_t* d_lengths = NULL;
  uint32_t* d_sucessors = NULL;
  uint32_t* d_box_length = NULL;
  uint32_t* d_num_boxes_axis = NULL;
  int32_t* d_grid_dimensions = NULL;

  cudaMalloc(&d_starts, starts->size() * sizeof(uint32_t));
  cudaMalloc(&d_lengths, lengths->size() * sizeof(uint16_t));
  cudaMalloc(&d_sucessors, successors->size() * sizeof(uint32_t));
  cudaMalloc(&d_box_length, sizeof(uint32_t));
  cudaMalloc(&d_num_boxes_axis, 3 * sizeof(uint32_t));
  cudaMalloc(&d_grid_dimensions, 3 * sizeof(int32_t));

  cudaMemcpy(d_starts, starts->data(), starts->size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lengths, lengths->data(), lengths->size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sucessors, successors->data(), successors->size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_box_length, &box_length, sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_num_boxes_axis, num_boxes_axis->data(), 3 * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_grid_dimensions, grid_dimensions->data(), 3 * sizeof(uint32_t), cudaMemcpyHostToDevice);

  // auto total_mem = 3*N*N*N*sizeof(REAL) + 3*N*N*N*sizeof(REAL) + N*N*N*sizeof(REAL);
  // std::cout << "total memory allocated = " << total_mem / (1024*1024) << " MB" << std::endl;
 
  // initialize
  const REAL space = 20;
  int i = 0;
  for (size_t x = 0; x < N; x++) {
    REAL x_pos = x * space;
    for (size_t y = 0; y < N; y++) {
      REAL y_pos = y * space;
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

  for (int t = 0; t < T; t++) {
    auto t1 = Clock::now();
    
    // Launch kernel
    int blockSize = 1024;
    int numBlocks = (N*N*N + blockSize - 1) / blockSize;
    collide<<<numBlocks, blockSize>>>(positions, diameters, N, d_starts, d_lengths, d_sucessors, d_box_length, d_num_boxes_axis, d_grid_dimensions, force);
   
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    auto t2 = Clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

    // remove("cuda.txt");
    // std::ofstream ofs("cuda.txt", std::ofstream::out);
    // for (int k = 0; k < N*N*N; k++) {
    //   ofs << force[3*k + 0] << ", " << force[3*k + 1] << ", " << force[3*k + 2] << std::endl;
    // }
    // ofs.close();

    REAL actual = compute_sum_cuda(force, N);
    if (are_same(actual, expected)) {
      std::cout << "Correct result! Because " << std::setprecision(15) << actual << " == " << expected << std::endl;
      clear_force(force, N);
      continue;    
    } else {
      std::cout << "Got result = " << actual << std::endl;
      std::cout << "Wrong result! Difference = " << fabs(actual - expected) << std::endl;
      std::cout << "Should be " << expected << std::endl;
      return 1;
    }
  }
 
  // Free memory
  cudaFree(positions);
  cudaFree(force);
  cudaFree(diameters);
  cudaFree(d_starts);
  cudaFree(d_lengths);
  cudaFree(d_box_length);
  cudaFree(d_sucessors);
  cudaFree(d_grid_dimensions);
  cudaFree(d_num_boxes_axis);

  return 0;
}
