#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "grid.h"

#define REAL float

typedef std::chrono::high_resolution_clock Clock;

float compute_sum(std::vector<std::array<REAL, 3>>& voa) {
  float sum = 0.0;

  for (auto& arr : voa) {
    sum += std::fabs(arr[0]);
    sum += std::fabs(arr[1]);
    sum += std::fabs(arr[2]);
  }
  return sum;
}

void clear_force_cpu(std::vector<std::array<float, 3>>* voa) {
  for (int i = 0; i < voa->size(); i++) {
    (*voa)[i][0] = 0;
    (*voa)[i][1] = 0;
    (*voa)[i][2] = 0;
  }
}

void calculate_collisions(const std::array<REAL, 3> &ref_mass_location,
                          REAL ref_diameter, REAL ref_iof_coefficient,
                          const std::array<REAL, 3> &nb_mass_location,
                          REAL nb_diameter, REAL nb_iof_coefficient,
                          std::array<REAL, 3> *result, size_t nidx) {
  auto c1 = ref_mass_location;
  REAL r1 = 0.5 * ref_diameter;
  auto c2 = nb_mass_location;
  REAL r2 = 0.5 * nb_diameter;
  // We take virtual bigger radii to have a distant interaction, to get a
  // desired density.
  REAL additional_radius = 10.0 * 0.15;
  r1 += additional_radius;
  r2 += additional_radius;
  // the 3 components of the vector c2 -> c1
  REAL comp1 = c1[0] - c2[0];
  REAL comp2 = c1[1] - c2[1];
  REAL comp3 = c1[2] - c2[2];
  REAL center_distance =
      std::sqrt(comp1 * comp1 + comp2 * comp2 + comp3 * comp3);

  // the overlap distance (how much one penetrates in the other)
  REAL delta = r1 + r2 - center_distance;

  if (delta < 0) {
    return;
  }
  // to avoid a division by 0 if the centers are (almost) at the same location
  if (center_distance < 0.00000001) {
    (*result)[0] += 42.0;
    (*result)[1] += 42.0;
    (*result)[2] += 42.0;
    return;
  }

  // the force itself
  REAL r = (r1 * r2) / (r1 + r2);
  REAL gamma = 1; // attraction coeff
  REAL k = 2;     // repulsion coeff
  REAL f = k * delta - gamma * std::sqrt(r * delta);

  REAL module = f / center_distance;
  std::array<REAL, 3> force2on1(
      {module * comp1, module * comp2, module * comp3});
  (*result)[0] += force2on1[0];
  (*result)[1] += force2on1[1];
  (*result)[2] += force2on1[2];
}

void FlushCache() {
  const int N = 100 * 1024 * 1024;
  char* tmp = (char*) malloc(100 * 1024 * 1024);
  for (int i = 0; i < N; i++) {
    tmp[i] = i;
  }
  delete tmp;
}


void cpu(std::vector<std::array<REAL, 3>>& positions, std::vector<std::array<REAL, 3>>* force, const std::vector<REAL>& diameters, Grid& g, size_t N) {
#pragma omp parallel for
  for (size_t i = 0; i < N * N * N; i++) {
    auto lambda = [&](size_t nidx) {
      calculate_collisions(positions[i], diameters[i], 0.15, positions[nidx], diameters[nidx], 0.15, &((*force)[i]), nidx);
    };
    g.ForEachNeighborWithinRadius(lambda, &positions, i, 1200);
  }
}

float calculate_expected(std::vector<std::array<REAL, 3>>& positions, std::vector<std::array<REAL, 3>>* force, const std::vector<REAL>& diameters, Grid& g, size_t N) {
  cpu(positions, force, diameters, g, N);
  return compute_sum(*force);
}
