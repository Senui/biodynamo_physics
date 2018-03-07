// kernel collide_kernel start 
// kernel stringified at 2018-02-08 11:57

#ifndef COLLIDE_KERNEL_H_
#define COLLIDE_KERNEL_H_

const char* const collide_kernel = "int3 get_box_coordinates(float3 pos, __constant int* grid_dimensions, uint box_length) {\n"
"  int3 box_coords;\n"
"  box_coords.x = (floor(pos.x) - grid_dimensions[0]) / box_length;\n"
"  box_coords.y = (floor(pos.y) - grid_dimensions[1]) / box_length;\n"
"  box_coords.z = (floor(pos.z) - grid_dimensions[2]) / box_length;\n"
"  return box_coords;\n"
"}\n"
"\n"
"uint get_box_id_2(int3 bc,__constant uint* num_boxes_axis) {\n"
"  return bc.z * num_boxes_axis[0]*num_boxes_axis[1] + bc.y * num_boxes_axis[0] + bc.x;\n"
"}\n"
"\n"
"uint get_box_id(float3 pos, __constant uint* num_boxes_axis, __constant int* grid_dimensions, uint box_length) {\n"
"  int3 box_coords = get_box_coordinates(pos, grid_dimensions, box_length);\n"
"  return get_box_id_2(box_coords, num_boxes_axis);\n"
"}\n"
"\n"
"void compute_force(__global float* positions, __global float* diameters, uint idx, uint nidx, __global float* result) {\n"
"  float r1 = 0.5 * diameters[idx];\n"
"  float r2 = 0.5 * diameters[nidx];\n"
"  // We take virtual bigger radii to have a distant interaction, to get a desired density.\n"
"  float additional_radius = 10.0 * 0.15;\n"
"  r1 += additional_radius;\n"
"  r2 += additional_radius;\n"
"\n"
"  float comp1 = positions[3*idx + 0] - positions[3*nidx + 0];\n"
"  float comp2 = positions[3*idx + 1] - positions[3*nidx + 1];\n"
"  float comp3 = positions[3*idx + 2] - positions[3*nidx + 2];\n"
"  float center_distance = sqrt(comp1 * comp1 + comp2 * comp2 + comp3 * comp3);\n"
"\n"
"  // the overlap distance (how much one penetrates in the other)\n"
"  float delta = r1 + r2 - center_distance;\n"
"\n"
"  if (delta < 0) {\n"
"    return;\n"
"  }\n"
"\n"
"  // to avoid a division by 0 if the centers are (almost) at the same location\n"
"  if (center_distance < 0.00000001) {\n"
"    result[3*idx + 0] += 42.0;\n"
"    result[3*idx + 1] += 42.0;\n"
"    result[3*idx + 2] += 42.0;\n"
"    return;\n"
"  }\n"
"\n"
"  // printf(\"Colliding cell [%d] and [%d]\\n\", idx, nidx);\n"
"  // printf(\"Delta for neighbor [%d] = %f\\n\", nidx, delta);\n"
"\n"
"  // the force itself\n"
"  float r = (r1 * r2) / (r1 + r2);\n"
"  float gamma = 1; // attraction coeff\n"
"  float k = 2;     // repulsion coeff\n"
"  float f = k * delta - gamma * sqrt(r * delta);\n"
"\n"
"  float module = f / center_distance;\n"
"  result[3*idx + 0] += module * comp1;\n"
"  result[3*idx + 1] += module * comp2;\n"
"  result[3*idx + 2] += module * comp3;\n"
"}\n"
"\n"
"\n"
"void default_force(__global float* positions,\n"
"                   __global float* diameters,\n"
"                   uint idx, uint start, ushort length,\n"
"                   __global uint* successors,\n"
"                   __global float* result) {\n"
"  uint nidx = start;\n"
"  // printf(\"start = %d \\n\", start);\n"
"  // printf(\"length = %d \\n\", length);\n"
"  // printf(\"nidx = %d \\n\\n\", nidx);\n"
"\n"
"  for (ushort nb = 0; nb < length; nb++) {\n"
"    // implement logic for within radius here\n"
"    if (nidx != idx) {\n"
"      // printf(\"%d\\n\", nidx);\n"
"      compute_force(positions, diameters, idx, nidx, result);\n"
"    }\n"
"    // traverse linked-list\n"
"    nidx = successors[nidx];\n"
"  }\n"
"  // printf(\"\\n\");\n"
"}\n"
"\n"
"__kernel void collide(__global float* positions,\n"
"                      __global float* diameters,\n"
"                      uint N,\n"
"                      __global uint* starts,\n"
"                      __global ushort* lengths,\n"
"                      __global uint* successors,\n"
"                      uint box_length,\n"
"                      __constant uint* num_boxes_axis,\n"
"                      __constant int* grid_dimensions,\n"
"                      __global float* result) {\n"
"  uint tidx = get_global_id(0);\n"
"  if (tidx < N * N * N) {\n"
"    // if (tidx == 0) {\n"
"      float3 pos;\n"
"      pos.x = positions[3*tidx + 0];\n"
"      pos.y = positions[3*tidx + 1];\n"
"      pos.z = positions[3*tidx + 2];\n"
"\n"
"      // printf(\"sucessors = \");\n"
"      // for (int i = 0; i < N*N*N; i++) {\n"
"        // printf(\"%u, \", successors[i]);\n"
"      // }\n"
"\n"
"      int3 box_coords = get_box_coordinates(pos, grid_dimensions, box_length);\n"
"      // printf(\"Colliding cell %d \\n\", tidx);\n"
"      // printf(\"pos %f, %f, %f \\n\", pos.x, pos.y, pos.z);\n"
"      // printf(\"Box %d, %d, %d \\n\", box_coords.x, box_coords.y, box_coords.z);\n"
"\n"
"      // Moore neighborhood\n"
"      for (int z = -1; z <= 1; z++) {\n"
"        for (int y = -1; y <= 1; y++) {\n"
"          for (int x = -1; x <= 1; x++) {\n"
"            uint bidx = get_box_id_2(box_coords + (int3)(x, y, z), num_boxes_axis);\n"
"            if (lengths[bidx] != 0) {\n"
"              // printf(\"Box %d \\n\", bidx);\n"
"              // printf(\"length = %d\\n\", lengths[bidx]);\n"
"              default_force(positions, diameters, tidx, starts[bidx], lengths[bidx], successors, result);\n"
"            }\n"
"          }\n"
"        }\n"
"      }\n"
"      // barrier(CLK_GLOBAL_MEM_FENCE);\n"
"    }\n"
"  // }\n"
"}\n"
"\n"
"__kernel void clear_force_opencl(__global float* result, uint N) {\n"
"  uint tidx = get_global_id(0);\n"
"  if (tidx < N * N * N) {\n"
"    result[3*tidx + 0] = 0;\n"
"    result[3*tidx + 1] = 0;\n"
"    result[3*tidx + 2] = 0;\n"
"  }\n"
"}\n"
;
// kernel collide_kernel end 

#endif  // COLLIDE_KERNEL_H_
