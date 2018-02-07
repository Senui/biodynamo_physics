#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <random>
#include <math.h>

int cuda_collide(std::vector<uint32_t>* starts,
                 std::vector<uint16_t>* lengths,
                 std::vector<uint32_t>* successors,
                 uint32_t box_length,
                 std::array<uint32_t, 3>* num_boxes_axis,
                 std::array<int32_t, 3>* grid_dimensions,
                 int N, int T, int diameter, float expected);
