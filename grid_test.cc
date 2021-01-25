#include "gpu.h"
#include "grid.h"
#include "helper.h"

#include "omp.h"

#include <chrono>
#include <fstream>

typedef std::chrono::high_resolution_clock Clock;

int main(int argc, char** argv) {
	// Create and initialize cell positions
	int N = std::stoi(argv[1]);
	std::vector<std::array<REAL, 3>> positions;
	initialize(positions, N);

	// If selected, sort the positions based on the Z-order curve (Morton)
	if (std::stoi(argv[2]) == 1) {
		auto t1 = Clock::now();
		morton_sort(&positions);
		auto t2 = Clock::now();
		std::cout << "Sorting = " 
			  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " ms" << std::endl;
	}

	// Perform spatial decomposition
	Grid& g = Grid::GetInstance();
	g.Initialize(&positions, 30);

	size_t* successors = g.GetSuccessors();
	Grid::GPUBox* gpu_boxes = new Grid::GPUBox[positions.size()];
	g.GetGPUBoxes(gpu_boxes);

	opencl(positions)

	// auto lambda = [&](size_t nidx, std::vector<size_t>* nidc) {
	// 	nidc->push_back(nidx);
	// };

	// std::vector<size_t> starts(positions.size());
	// std::vector<size_t> indices;
	// starts[0] = 0;

	// auto t1 = Clock::now();
	// for (size_t i = 0; i < positions.size(); i++) {
	// 	std::vector<size_t> nidc;
	// 	g.ForEachNeighborWithinRadius(lambda, &positions, i, 900, &nidc);
	// 	starts[i + 1] = starts[i] + nidc.size();
	// 	indices.insert(indices.end(), nidc.begin(), nidc.end());
	// }
	// auto t2 = Clock::now();

	std::cout << "For-all-neighbors operation = " 
			  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " ms" << std::endl;

	// std::cout << std::accumulate(all_cs.begin(), all_cs.end(), 0) << " neighbor operations" << std::endl;          

	// for (size_t i = 0; i < starts.size(); i++) {
	// 	std::cout << "Neighbors of cell [" << i << "] = [";
	// 	size_t end = (i == (starts.size() - 1)) ? indices.size() : starts[i + 1];
	// 	for (size_t j = starts[i]; j < end; j++) {
	// 		std::cout << indices[j];
	// 		if (j != end - 1) {
	// 			std::cout << ", ";
	// 		}
	// 	}
	// 	std::cout << "]" << std::endl;
	// }

	// for (size_t i = 0; i < positions.size(); i++) {
	// 	std::cout << "Cell [" << i << "] = ["
	// 			  << positions[i][0]/20 << ", "
	// 			  << positions[i][1]/20 << ", "
	// 			  << positions[i][2]/20 << "]\n";
	// }

	// remove("neighbors.txt");
 // std::ofstream ofs("neighbors.txt", std::ofstream::out);
	// for (size_t i = 0; i < starts.size(); i++) {
 //    	size_t end = (i == (starts.size() - 1)) ? nidc.size() : starts[i + 1];
 //    	for (size_t j = starts[i]; j < end; j++) {
 //    		ofs << nidc[j];
	// 		if (j != end - 1) {
	// 			ofs << ", ";
	// 		}
 //    	}
	// 	ofs << std::endl;
	// }
	// ofs.close();

	return 0;
}
