#define STRINGIFY(src) #src

static const std::string source = STRINGIFY(
__kernel void collide(
       __global float* positions,
       __global float* diameters,
       __global float* result,
       __global int* nidc,
       int N,
       int cpc) {
  size_t i = get_global_id(0);
  if (i < N*N*N) {
    for (int nb = 0; nb < cpc; nb++) {
      float r1 = 0.5 * diameters[i];
      float r2 = 0.5 * diameters[nidc[cpc * i + nb]];
      // We take virtual bigger radii to have a distant interaction, to get a
      // desired density.
      float additional_radius = 10.0 * 0.15;
      r1 += additional_radius;
      r2 += additional_radius;

      float comp1 = positions[3*i + 0] - positions[3 * nidc[cpc * i + nb] + 0];
      float comp2 = positions[3*i + 1] - positions[3 * nidc[cpc * i + nb] + 1];
      float comp3 = positions[3*i + 2] - positions[3 * nidc[cpc * i + nb] + 2];
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
      barrier(CLK_GLOBAL_MEM_FENCE);
  }
});
