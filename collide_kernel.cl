int3 get_box_coordinates(float3 pos, __constant int* grid_dimensions, uint box_length) {
  int3 box_coords;
  box_coords.x = (floor(pos.x) - grid_dimensions[0]) / box_length;
  box_coords.y = (floor(pos.y) - grid_dimensions[1]) / box_length;
  box_coords.z = (floor(pos.z) - grid_dimensions[2]) / box_length;
  return box_coords;
}

uint get_box_id_2(int3 bc,__constant uint* num_boxes_axis) {
  return bc.z * num_boxes_axis[0]*num_boxes_axis[1] + bc.y * num_boxes_axis[0] + bc.x;
}

uint get_box_id(float3 pos, __constant uint* num_boxes_axis, __constant int* grid_dimensions, uint box_length) {
  int3 box_coords = get_box_coordinates(pos, grid_dimensions, box_length);
  return get_box_id_2(box_coords, num_boxes_axis);
}

void compute_force(__global float* positions, __global float* diameters, uint idx, uint nidx, __global float* result) {
  float r1 = 0.5 * diameters[idx];
  float r2 = 0.5 * diameters[nidx];
  // We take virtual bigger radii to have a distant interaction, to get a desired density.
  float additional_radius = 10.0 * 0.15;
  r1 += additional_radius;
  r2 += additional_radius;

  float comp1 = positions[3*idx + 0] - positions[3*nidx + 0];
  float comp2 = positions[3*idx + 1] - positions[3*nidx + 1];
  float comp3 = positions[3*idx + 2] - positions[3*nidx + 2];
  float center_distance = sqrt(comp1 * comp1 + comp2 * comp2 + comp3 * comp3);

  // the overlap distance (how much one penetrates in the other)
  float delta = r1 + r2 - center_distance;

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

  // printf(\"Colliding cell [%d] and [%d]\\n\", idx, nidx);
  // printf(\"Delta for neighbor [%d] = %f\\n\", nidx, delta);

  // the force itself
  float r = (r1 * r2) / (r1 + r2);
  float gamma = 1; // attraction coeff
  float k = 2;     // repulsion coeff
  float f = k * delta - gamma * sqrt(r * delta);

  float module = f / center_distance;
  result[3*idx + 0] += module * comp1;
  result[3*idx + 1] += module * comp2;
  result[3*idx + 2] += module * comp3;
}


void default_force(__global float* positions,
                   __global float* diameters,
                   uint idx, uint start, ushort length,
                   __global uint* successors,
                   __global float* result) {
  uint nidx = start;
  // printf(\"start = %d \\n\", start);
  // printf(\"length = %d \\n\", length);
  // printf(\"nidx = %d \\n\\n\", nidx);

  for (ushort nb = 0; nb < length; nb++) {
    // implement logic for within radius here
    if (nidx != idx) {
      // printf(\"%d\\n\", nidx);
      compute_force(positions, diameters, idx, nidx, result);
    }
    // traverse linked-list
    nidx = successors[nidx];
  }
  // printf(\"\\n\");
}

__kernel void collide(__global float* positions,
                      __global float* diameters,
                      uint N,
                      __global uint* starts,
                      __global ushort* lengths,
                      __global uint* successors,
                      uint box_length,
                      __constant uint* num_boxes_axis,
                      __constant int* grid_dimensions,
                      __global float* result) {
  uint tidx = get_global_id(0);
  if (tidx < N * N * N) {
    // if (tidx == 0) {
      float3 pos;
      pos.x = positions[3*tidx + 0];
      pos.y = positions[3*tidx + 1];
      pos.z = positions[3*tidx + 2];

      // printf(\"sucessors = \");
      // for (int i = 0; i < N*N*N; i++) {
        // printf(\"%u, \", successors[i]);
      // }

      int3 box_coords = get_box_coordinates(pos, grid_dimensions, box_length);
      // printf(\"Colliding cell %d \\n\", tidx);
      // printf(\"pos %f, %f, %f \\n\", pos.x, pos.y, pos.z);
      // printf(\"Box %d, %d, %d \\n\", box_coords.x, box_coords.y, box_coords.z);

      // Moore neighborhood
      for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
          for (int x = -1; x <= 1; x++) {
            uint bidx = get_box_id_2(box_coords + (int3)(x, y, z), num_boxes_axis);
            if (lengths[bidx] != 0) {
              // printf(\"Box %d \\n\", bidx);
              // printf(\"length = %d\\n\", lengths[bidx]);
              default_force(positions, diameters, tidx, starts[bidx], lengths[bidx], successors, result);
            }
          }
        }
      }
      // barrier(CLK_GLOBAL_MEM_FENCE);
    }
  // }
}

__kernel void clear_force_opencl(__global float* result, uint N) {
  uint tidx = get_global_id(0);
  if (tidx < N * N * N) {
    result[3*tidx + 0] = 0;
    result[3*tidx + 1] = 0;
    result[3*tidx + 2] = 0;
  }
}
