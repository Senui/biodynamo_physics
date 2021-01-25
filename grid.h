#ifndef GRID_H_
#define GRID_H_

#include <assert.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

using std::array;
using std::fmod;

#define REAL float

/// Vector with fixed number of elements == Array with push_back function that
/// keeps track of its size
/// NB: No bounds checking. Do not push_back more often than the number of
/// maximum elements given by the template parameter N
template <typename T, std::size_t N>
class FixedSizeVector {
 public:
  size_t size() const { return size_; }  // NOLINT

  const T& operator[](size_t idx) const { return data_[idx]; }

  T& operator[](size_t idx) { return data_[idx]; }

  FixedSizeVector& operator++() {
#pragma omp simd
    for (size_t i = 0; i < N; i++) {
      ++data_[i];
    }
    return *this;
  }

  void clear() { size_ = 0; }  // NOLINT

  void push_back(const T& value) {  // NOLINT
    assert(size_ < N);
    data_[size_++] = value;
  }

  const T* begin() const { return &(data_[0]); }    // NOLINT
  const T* end() const { return &(data_[size_]); }  // NOLINT
  T* begin() { return &(data_[0]); }                // NOLINT
  T* end() { return &(data_[size_]); }              // NOLINT

 private:
  T data_[N];
  std::size_t size_ = 0;
};

/// A class that represents Cartesian 3D grid
class Grid {
 public:
  /// A single unit cube of the grid
  struct Box {
    /// start value of the linked list of simulation objects inside this box.
    /// Next element can be found at `successors_[start_]`
    std::atomic<uint32_t> start_;
    /// length of the linked list (i.e. number of simulation objects)
    std::atomic<uint16_t> length_;

    Box() : start_(std::numeric_limits<uint32_t>::max()), length_(0) {}
    /// Copy Constructor required for boxes_.resize()
    /// Since box values will be overwritten afterwards it forwards to the
    /// default ctor
    Box(const Box& other) : Box() {}
    /// Required for boxes_.resize
    /// Since box values will be overwritten afterwards, implementation is
    /// missing
    const Box& operator=(const Box& other) const { return *this; }

    bool IsEmpty() const { return length_ == 0; }

    /// @brief      Adds a simulation object to this box
    ///
    /// @param[in]  obj_id       The object's identifier
    /// @param      successors   The successors
    ///
    /// @tparam     TSuccessors  Type of successors
    ///
    template <typename TSimulationObjectVector>
    void AddObject(uint32_t obj_id, TSimulationObjectVector* successors) {
      length_++;
      auto old_start = std::atomic_exchange(&start_, obj_id);
      if (old_start != std::numeric_limits<uint32_t>::max()) {
        (*successors)[obj_id] = old_start;
      }
    }

    /// An iterator that iterates over the cells in this box
    struct Iterator {
      Iterator(Grid* grid, const Box* box)
          : grid_(grid),
            current_value_(box->start_),
            countdown_(box->length_) {}

      bool IsAtEnd() { return countdown_ <= 0; }

      Iterator& operator++() {
        countdown_--;
        if (countdown_ > 0) {
          current_value_ = grid_->successors_[current_value_];
        }
        return *this;
      }

      size_t operator*() const { return current_value_; }

      /// Pointer to the neighbor grid; for accessing the successor_ list
      Grid* grid_;
      /// The current simulation object to be considered
      size_t current_value_;
      /// The remain number of simulation objects to consider
      int countdown_ = 0;
    };

    Iterator begin() const {  // NOLINT
      return Iterator(&(Grid::GetInstance()), this);
    }
  };

  struct GridData {
    cl_uint box_length_;
    cl_uint num_boxes_axis_[3];
    // only the minima
    cl_uint grid_dimensions_[3];
  };

  /// An iterator that iterates over the boxes in this grid
  struct NeighborIterator {
    explicit NeighborIterator(
        const FixedSizeVector<const Box*, 27>& neighbor_boxes)
        : neighbor_boxes_(neighbor_boxes),
          // start iterator from box 0
          box_iterator_(neighbor_boxes_[0]->begin()) {
      // if first box is empty
      if (neighbor_boxes_[0]->IsEmpty()) {
        ForwardToNonEmptyBox();
      }
    }

    bool IsAtEnd() const { return is_end_; }

    size_t operator*() const { return *box_iterator_; }

    /// Version where empty neighbor boxes are allowed
    NeighborIterator& operator++() {
      ++box_iterator_;
      // if iterator of current box has come to an end, continue with next box
      if (box_iterator_.IsAtEnd()) {
        return ForwardToNonEmptyBox();
      }
      return *this;
    }

   private:
    /// The 27 neighbor boxes that will be searched for simulation objects
    const FixedSizeVector<const Box*, 27>& neighbor_boxes_;
    /// The box that shall be considered to iterate over for finding simulation
    /// objects
    typename Box::Iterator box_iterator_;
    /// The id of the box to be considered (i.e. value between 0 - 26)
    uint16_t box_idx_ = 0;
    /// Flag to indicate that all the neighbor boxes have been searched through
    bool is_end_ = false;

    /// Forwards the iterator to the next non empty box and returns itself
    /// If there are no non empty boxes is_end_ is set to true
    NeighborIterator& ForwardToNonEmptyBox() {
      // increment box id until non empty box has been found
      while (++box_idx_ < neighbor_boxes_.size()) {
        // box is empty or uninitialized (padding box) -> continue
        if (neighbor_boxes_[box_idx_]->IsEmpty()) {
          continue;
        }
        // a non-empty box has been found
        box_iterator_ = neighbor_boxes_[box_idx_]->begin();
        return *this;
      }
      // all remaining boxes have been empty; reached end
      is_end_ = true;
      return *this;
    }
  };

  /// Enum that determines the degree of adjacency in search neighbor boxes
  //  todo(ahmad): currently only kHigh is supported (hardcoded 26 several
  //  places)
  enum Adjacency {
    kLow,    /**< The closest 8  neighboring boxes */
    kMedium, /**< The closest 18  neighboring boxes */
    kHigh    /**< The closest 26  neighboring boxes */
  };

  Grid() {}

  Grid(Grid const&) = delete;
  void operator=(Grid const&) = delete;

  /// @brief      Initialize the grid with the given simulation objects
  /// @param[in]  adjacency    The adjacency (see #Adjacency)
  void Initialize(std::vector<std::array<REAL, 3>>* positions, uint32_t box_length, Adjacency adjacency = kHigh) {
    adjacency_ = adjacency;

    int32_t inf = std::numeric_limits<int32_t>::max();

    UpdateGrid(positions, box_length);
    initialized_ = true;
  }

  virtual ~Grid() {}

  /// Gets the singleton instance
  static Grid& GetInstance() {
    static Grid kGrid;
    return kGrid;
  }

  /// Clears the grid
  void ClearGrid() {
    boxes_.clear();
    box_length_ = 1;
    largest_object_size_ = 0;
    num_boxes_axis_ = {{0}};
    num_boxes_xy_ = 0;
    int32_t inf = std::numeric_limits<int32_t>::max();
    grid_dimensions_ = {inf, -inf, inf, -inf, inf, -inf};
    successors_.clear();
    has_grown_ = false;
  }

  /// Updates the grid, as simulation objects may have moved, added or deleted
  void UpdateGrid(std::vector<std::array<REAL, 3>>* positions, uint32_t box_length) {
    ClearGrid();

    auto inf = std::numeric_limits<REAL>::max();
    array<REAL, 6> tmp_dim = {{inf, -inf, inf, -inf, inf, -inf}};
    CalculateGridDimensions(positions, &tmp_dim);
    RoundOffGridDimensions(tmp_dim);

    box_length_ = box_length;

    for (int i = 0; i < 3; i++) {
      int dimension_length =
          grid_dimensions_[2 * i + 1] - grid_dimensions_[2 * i];
      int r = dimension_length % box_length_;
      // If the grid is not perfectly divisible along each dimension by the
      // resolution, extend the grid so that it is
      if (r != 0) {
        // std::abs for the case that box_length_ > dimension_length
        grid_dimensions_[2 * i + 1] += (box_length_ - r);
      } else {
        // Else extend the grid dimension with one row, because the outmost
        // object lies exactly on the border
        grid_dimensions_[2 * i + 1] += box_length_;
      }
    }

    // Pad the grid to avoid out of bounds check when searching neighbors
    for (int i = 0; i < 3; i++) {
      grid_dimensions_[2 * i] -= box_length_;
      grid_dimensions_[2 * i + 1] += box_length_;
    }

    // Calculate how many boxes fit along each dimension
    for (int i = 0; i < 3; i++) {
      int dimension_length =
          grid_dimensions_[2 * i + 1] - grid_dimensions_[2 * i];
      assert((dimension_length % box_length_ == 0) &&
             "The grid dimensions are not a multiple of its box length");
      num_boxes_axis_[i] = dimension_length / box_length_;
    }

    num_boxes_xy_ = num_boxes_axis_[0] * num_boxes_axis_[1];
    auto total_num_boxes = num_boxes_xy_ * num_boxes_axis_[2];

    if (boxes_.size() != total_num_boxes) {
      boxes_.resize(total_num_boxes, Box());
    }

    successors_.resize(positions->size());

    // Assign simulation objects to boxes
    for (uint32_t id = 0; id < positions->size(); id++) {
      const auto& position = (*positions)[id];
      auto idx = this->GetBoxIndex(position);
      auto box = this->GetBoxPointer(idx);
      box->AddObject(id, &successors_);
    }
  }

  /// Calculates what the grid dimensions need to be in order to contain all the
  /// simulation objects
  void CalculateGridDimensions(std::vector<std::array<REAL, 3>>* positions, array<REAL, 6>* ret_grid_dimensions) {
    // const auto max_threads = omp_get_max_threads();

    // std::vector<std::array<REAL, 6>*> all_grid_dimensions(max_threads,
                                                            // nullptr);
// #pragma omp parallel
//     {
//       auto thread_id = omp_get_thread_num();
//       auto* grid_dimensions = new std::array<REAL, 6>;
//       auto inf = std::numeric_limits<REAL>::max();
//       *grid_dimensions = {{inf, -inf,
//                            inf, -inf,
//                            inf, -inf}};
//       all_grid_dimensions[thread_id] = grid_dimensions;

// #pragma omp for
      for (size_t i = 0; i < positions->size(); i++) {
        const auto& position = (*positions)[i];
        for (size_t j = 0; j < 3; j++) {
          if (position[j] < (*ret_grid_dimensions)[2 * j]) {
            (*ret_grid_dimensions)[2 * j] = position[j];
          }
          if (position[j] > (*ret_grid_dimensions)[2 * j + 1]) {
            (*ret_grid_dimensions)[2 * j + 1] = position[j];
          }
        }
      }

// #pragma omp master
//       {
//         for (int i = 0; i < max_threads; i++) {
//           for (size_t j = 0; j < 3; j++) {
//             if ((*all_grid_dimensions[i])[2 * j] <
//                 (*ret_grid_dimensions)[2 * j]) {
//               (*ret_grid_dimensions)[2 * j] = (*all_grid_dimensions[i])[2 * j];
//             }
//             if ((*all_grid_dimensions[i])[2 * j + 1] >
//                 (*ret_grid_dimensions)[2 * j + 1]) {
//               (*ret_grid_dimensions)[2 * j + 1] =
//                   (*all_grid_dimensions[i])[2 * j + 1];
//             }
//           }
//         }
//       }
//     }

//     for (auto element : all_grid_dimensions) {
//       delete element;
//     }
  }

  void RoundOffGridDimensions(const array<REAL, 6>& grid_dimensions) {
    assert(grid_dimensions_[0] > -9.999999999);
    assert(grid_dimensions_[2] > -9.999999999);
    assert(grid_dimensions_[4] > -9.999999999);
    assert(grid_dimensions_[1] < 80);
    assert(grid_dimensions_[3] < 80);
    assert(grid_dimensions_[5] < 80);
    grid_dimensions_[0] = floor(grid_dimensions[0]);
    grid_dimensions_[2] = floor(grid_dimensions[2]);
    grid_dimensions_[4] = floor(grid_dimensions[4]);
    grid_dimensions_[1] = ceil(grid_dimensions[1]);
    grid_dimensions_[3] = ceil(grid_dimensions[3]);
    grid_dimensions_[5] = ceil(grid_dimensions[5]);
  }

  /// @brief      Calculates the squared euclidian distance between two points
  ///             in 3D
  ///
  /// @param[in]  pos1  Position of the first point
  /// @param[in]  pos2  Position of the second point
  ///
  /// @return     The distance between the two points
  ///
  inline REAL SquaredEuclideanDistance(
      const std::array<REAL, 3>& pos1,
      const std::array<REAL, 3>& pos2) const {
    const REAL dx = pos2[0] - pos1[0];
    const REAL dy = pos2[1] - pos1[1];
    const REAL dz = pos2[2] - pos1[2];
    return (dx * dx + dy * dy + dz * dz);
  }

  /// @brief      Applies the given lambda to each neighbor or the specified
  ///             simulation object
  ///
  /// @param[in]  lambda  The operation as a lambda
  /// @param      query   The query object
  /// @param      simulation_object_id
  /// @param[in]  squared_radius  The search radius squared
  ///
  /// @tparam     Lambda      The type of the lambda operation
  /// @tparam     SO          The type of the simulation object
  ///
  template <typename Lambda>
  void ForEachNeighborWithinRadius(const Lambda& lambda, std::vector<std::array<REAL, 3>>* positions, 
                                   size_t simulation_object_id,
                                   REAL squared_radius) {
    const auto& position = (*positions)[simulation_object_id];

    FixedSizeVector<const Box*, 27> neighbor_boxes;
    GetMooreBoxes(&neighbor_boxes, GetBoxIndex(position));

    NeighborIterator ni(neighbor_boxes);
    while (!ni.IsAtEnd()) {
      // Do something with neighbor object
      size_t neighbor_id = *ni;
      if (neighbor_id != simulation_object_id) {
        const auto& neighbor_position = (*positions)[neighbor_id];
        if (this->SquaredEuclideanDistance(position, neighbor_position) <
            squared_radius) {
          lambda(neighbor_id);
        }
      }
      ++ni;
    }
  }

  /// @brief      Return the box index in the one dimensional array of the box
  ///             that contains the position
  ///
  /// @param[in]  position  The position of the object
  ///
  /// @return     The box index.
  ///
  size_t GetBoxIndex(const array<REAL, 3>& position) const {
    array<uint32_t, 3> box_coord;
    box_coord[0] = (floor(position[0]) - grid_dimensions_[0]) / box_length_;
    box_coord[1] = (floor(position[1]) - grid_dimensions_[2]) / box_length_;
    box_coord[2] = (floor(position[2]) - grid_dimensions_[4]) / box_length_;

    return GetBoxIndex(box_coord);
  }

  /// Gets the size of the largest object in the grid
  REAL GetLargestObjectSize() const { return largest_object_size_; }

  array<int32_t, 6>& GetDimensions() { return grid_dimensions_; }

  array<uint32_t, 3>& GetNumBoxes() { return num_boxes_axis_; }

  uint32_t GetBoxLength() { return box_length_; }

  bool HasGrown() { return has_grown_; }

  std::array<uint64_t, 3> GetBoxCoordinates(size_t box_idx) const {
    std::array<uint64_t, 3> box_coord;
    box_coord[2] = box_idx / num_boxes_xy_;
    auto remainder = box_idx % num_boxes_xy_;
    box_coord[1] = remainder / num_boxes_axis_[0];
    box_coord[0] = remainder % num_boxes_axis_[0];
    return box_coord;
  }

  bool IsInitialized() { return initialized_; }

  template <typename T_UINT32>
  void GetSuccessors(std::vector<T_UINT32>* successors) {
    int i = 0;
    for (auto& s : successors_) {
      (*successors)[i] = s;
      i++;
    }
  }

  template <typename T_UINT32, typename T_UINT16>
  void GetGPUBoxData(std::vector<T_UINT32>* gpu_starts, std::vector<T_UINT16>* gpu_lengths) {
    gpu_starts->resize(boxes_.size());
    gpu_lengths->resize(boxes_.size());
    size_t i = 0;
    for (auto& box : boxes_) {
      (*gpu_starts)[i] = box.start_;
      (*gpu_lengths)[i] = box.length_;
      i++;
    }
  }

  template <typename T_UINT32, typename T_INT32>
  void GetGridData(T_UINT32* box_length, std::array<T_UINT32, 3>& num_boxes_axis, std::array<T_INT32, 3>& grid_dimensions) {
    box_length[0] = box_length_;
    num_boxes_axis[0] = num_boxes_axis_[0];
    num_boxes_axis[1] = num_boxes_axis_[1];
    num_boxes_axis[2] = num_boxes_axis_[2];
    grid_dimensions[0] = grid_dimensions_[0];
    grid_dimensions[1] = grid_dimensions_[2];
    grid_dimensions[2] = grid_dimensions_[4];
  }

  void PrintSuccessors() {
    std::cout << "successors_ = [";
    for (auto& s : successors_) {
      std::cout << s << ", ";
    }
    std::cout << "]" << std::endl;
  }

 private:
  /// The vector containing all the boxes in the grid
  std::vector<Box> boxes_;
  /// Length of a Box
  uint32_t box_length_ = 1;
  /// Stores the number of boxes for each axis
  array<uint32_t, 3> num_boxes_axis_ = {{0}};
  /// Number of boxes in the xy plane (=num_boxes_axis_[0] * num_boxes_axis_[1])
  size_t num_boxes_xy_ = 0;
  /// Implements linked list - array index = key, value: next element
  ///
  ///     // Usage
  ///     SoHandle current_element = ...;
  ///     SoHandle next_element = successors_[current_element];
  std::vector<uint32_t> successors_;
  /// Determines which boxes to search neighbors in (see enum Adjacency)
  Adjacency adjacency_;
  /// The size of the largest object in the simulation
  REAL largest_object_size_ = 0;
  /// Cube which contains all simulation objects
  /// {x_min, x_max, y_min, y_max, z_min, z_max}
  std::array<int32_t, 6> grid_dimensions_;
  // Flag to indicate that the grid dimensions have increased
  bool has_grown_ = false;
  /// Flag to indicate if the grid has been initialized or not
  bool initialized_ = false;

  /// @brief      Gets the Moore (i.e adjacent) boxes of the query box
  ///
  /// @param      neighbor_boxes  The neighbor boxes
  /// @param[in]  box_idx         The query box
  ///
  void GetMooreBoxes(FixedSizeVector<const Box*, 27>* neighbor_boxes,
                     size_t box_idx) const {
    neighbor_boxes->push_back(GetBoxPointer(box_idx));

    // Adjacent 6 (top, down, left, right, front and back)
    if (adjacency_ >= kLow) {
      neighbor_boxes->push_back(GetBoxPointer(box_idx - num_boxes_xy_));
      neighbor_boxes->push_back(GetBoxPointer(box_idx + num_boxes_xy_));
      neighbor_boxes->push_back(GetBoxPointer(box_idx - num_boxes_axis_[0]));
      neighbor_boxes->push_back(GetBoxPointer(box_idx + num_boxes_axis_[0]));
      neighbor_boxes->push_back(GetBoxPointer(box_idx - 1));
      neighbor_boxes->push_back(GetBoxPointer(box_idx + 1));
    }

    // Adjacent 12
    if (adjacency_ >= kMedium) {
      neighbor_boxes->push_back(
          GetBoxPointer(box_idx - num_boxes_xy_ - num_boxes_axis_[0]));
      neighbor_boxes->push_back(GetBoxPointer(box_idx - num_boxes_xy_ - 1));
      neighbor_boxes->push_back(
          GetBoxPointer(box_idx - num_boxes_axis_[0] - 1));
      neighbor_boxes->push_back(
          GetBoxPointer(box_idx + num_boxes_xy_ - num_boxes_axis_[0]));
      neighbor_boxes->push_back(GetBoxPointer(box_idx + num_boxes_xy_ - 1));
      neighbor_boxes->push_back(
          GetBoxPointer(box_idx + num_boxes_axis_[0] - 1));
      neighbor_boxes->push_back(
          GetBoxPointer(box_idx - num_boxes_xy_ + num_boxes_axis_[0]));
      neighbor_boxes->push_back(GetBoxPointer(box_idx - num_boxes_xy_ + 1));
      neighbor_boxes->push_back(
          GetBoxPointer(box_idx - num_boxes_axis_[0] + 1));
      neighbor_boxes->push_back(
          GetBoxPointer(box_idx + num_boxes_xy_ + num_boxes_axis_[0]));
      neighbor_boxes->push_back(GetBoxPointer(box_idx + num_boxes_xy_ + 1));
      neighbor_boxes->push_back(
          GetBoxPointer(box_idx + num_boxes_axis_[0] + 1));
    }

    // Adjacent 8
    if (adjacency_ >= kHigh) {
      neighbor_boxes->push_back(
          GetBoxPointer(box_idx - num_boxes_xy_ - num_boxes_axis_[0] - 1));
      neighbor_boxes->push_back(
          GetBoxPointer(box_idx - num_boxes_xy_ - num_boxes_axis_[0] + 1));
      neighbor_boxes->push_back(
          GetBoxPointer(box_idx - num_boxes_xy_ + num_boxes_axis_[0] - 1));
      neighbor_boxes->push_back(
          GetBoxPointer(box_idx - num_boxes_xy_ + num_boxes_axis_[0] + 1));
      neighbor_boxes->push_back(
          GetBoxPointer(box_idx + num_boxes_xy_ - num_boxes_axis_[0] - 1));
      neighbor_boxes->push_back(
          GetBoxPointer(box_idx + num_boxes_xy_ - num_boxes_axis_[0] + 1));
      neighbor_boxes->push_back(
          GetBoxPointer(box_idx + num_boxes_xy_ + num_boxes_axis_[0] - 1));
      neighbor_boxes->push_back(
          GetBoxPointer(box_idx + num_boxes_xy_ + num_boxes_axis_[0] + 1));
    }
  }

  /// @brief      Gets the pointer to the box with the given index
  ///
  /// @param[in]  index  The index of the box
  ///
  /// @return     The pointer to the box
  ///
  const Box* GetBoxPointer(size_t index) const { return &(boxes_[index]); }

  /// @brief      Gets the pointer to the box with the given index
  ///
  /// @param[in]  index  The index of the box
  ///
  /// @return     The pointer to the box
  ///
  Box* GetBoxPointer(size_t index) { return &(boxes_[index]); }

  /// Returns the box index in the one dimensional array based on box
  /// coordinates in space
  ///
  /// @param      box_coord  box coordinates in space (x, y, z)
  ///
  /// @return     The box index.
  ///
  size_t GetBoxIndex(const array<uint32_t, 3>& box_coord) const {
    return box_coord[2] * num_boxes_xy_ + box_coord[1] * num_boxes_axis_[0] +
           box_coord[0];
  }
};

#endif  // GRID_H_
