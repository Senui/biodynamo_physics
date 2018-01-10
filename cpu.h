#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#define REAL float

typedef std::chrono::high_resolution_clock Clock;

void calculate_collisions(const std::array<REAL, 3> &ref_mass_location,
                          REAL ref_diameter, REAL ref_iof_coefficient,
                          const std::array<REAL, 3> &nb_mass_location,
                          REAL nb_diameter, REAL nb_iof_coefficient,
                          std::array<REAL, 3> *result) {
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
    *result = {0.0, 0.0, 0.0};
    return;
  }
  // to avoid a division by 0 if the centers are (almost) at the same location
  if (center_distance < 0.00000001) {
    *result = {42.0, 42.0, 42.0};
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
  *result = force2on1;
  // printf("center_distance = %f, module = %f, f = %f, delta = %f, r1 = %f, r2 = %f", center_distance, module, f, delta, r1, r2);
}

void cpu(const std::vector<std::array<REAL, 3>>& positions, std::vector<std::array<REAL, 3>>* force, const std::vector<REAL>& diameters, const std::vector<int>& nidc, size_t N, int T, int cpc) {
	
	for (int t = 0; t < T; t++) {
	#pragma omp parallel for
	  for (size_t i = 0; i < N * N * N; i++) {
	    for (int nb = 0; nb < cpc; nb++) {
	      // std::cout << i << " vs " << nidc[cpc*i + nb] << std::endl;
        // printf("Colliding cell %d (@{%f, %f, %f}) and %d (@{%f, %f, %f}), ", i, positions[i][0], positions[i][1],positions[i][2], nidc[cpc * i + nb], positions[nidc[cpc * i + nb]][0], positions[nidc[cpc * i + nb]][1], positions[nidc[cpc * i + nb]][2]);
	      calculate_collisions(
	          positions[i], diameters[i], 0.15, positions[nidc[cpc * i + nb]],
	          diameters[nidc[cpc * i + nb]], 0.15, &((*force)[i]));
         // printf("force = [%f, %f, %f]\n", force[i][0], force[i][1], force[i][2]);
	    }
	  }
	}
}