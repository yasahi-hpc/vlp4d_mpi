#ifndef __VIEW_HPP__
#define __VIEW_HPP__

#include <vector>
#include <array>
#include <numeric>
#include <execution>
#include <algorithm>
#include "ViewLayout.hpp"

template <size_t ND>
using shape_nd = std::array<int, ND>;

/*
 * Helper class to convert flatten 1D index to multi-dimensional indices
 */
template <size_t ND, Layout LayoutType = Layout::LayoutLeft>
struct Coord {
  int strides_[ND];
  int offsets_[ND];
  Coord() = delete;

  Coord(const shape_nd<ND>& strides)
    : strides_ {0}, offsets_ {0} {
    for(int i=0; i<ND; i++) {
      strides_[i] = strides[i];
    }
  }

  Coord(const shape_nd<ND>& strides, const shape_nd<ND>& offsets)
    : strides_ {0}, offsets_ {0} {
    for(int i=0; i<ND; i++) {
      strides_[i] = strides[i];
      offsets_[i] = offsets[i];
    }
  }

  template <typename... I>
  Coord(I... indices) 
    : offsets_ {0} {
    int indices_tmp[ND] = {indices...};
    for(int i=0; i<ND; i++) {
      strides_[i] = indices_tmp[i];
    }
  }

  void to_coord(const int idx, int *ptr_idx) const noexcept {
    to_coord_(idx, ptr_idx);
  }

  using layout_ = std::integral_constant<Layout, LayoutType>;
  using dim_    = std::integral_constant<size_t, ND>;

  // For LayoutLeft
  template <size_t NDIM = ND, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutLeft>>::value &&
                                 std::is_same<std::integral_constant<size_t, NDIM>, std::integral_constant<size_t, 1>>::value, void>::type
  to_coord_(const int idx, int *ptr_idx) const noexcept {
    ptr_idx[0] = idx + offsets_[0];
  } 

  template <size_t NDIM = ND, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutLeft>>::value &&
                                 std::is_same<std::integral_constant<size_t, NDIM>, std::integral_constant<size_t, 2>>::value, void>::type
  to_coord_(const int idx, int *ptr_idx) const noexcept {
    int j0 = idx%strides_[0], j1 = idx/strides_[0];
    ptr_idx[0] = j0 + offsets_[0];
    ptr_idx[1] = j1 + offsets_[1];
  }

  template <size_t NDIM = ND, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutLeft>>::value &&
                                 std::is_same<std::integral_constant<size_t, NDIM>, std::integral_constant<size_t, 3>>::value, void>::type
  to_coord_(const int idx, int *ptr_idx) const noexcept {
    int j0 = idx%strides_[0], j12 = idx/strides_[0];
    int j1 = j12%strides_[1], j2 = j12/strides_[1]; 
    ptr_idx[0] = j0 + offsets_[0];
    ptr_idx[1] = j1 + offsets_[1];
    ptr_idx[2] = j2 + offsets_[2];
  }

  template <size_t NDIM = ND, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutLeft>>::value &&
                                 std::is_same<std::integral_constant<size_t, NDIM>, std::integral_constant<size_t, 4>>::value, void>::type
  to_coord_(const int idx, int *ptr_idx) const noexcept {
    int j0 = idx%strides_[0], j123 = idx/strides_[0];
    int j1 = j123%strides_[1], j23 = j123/strides_[1]; 
    int j2 = j23%strides_[2], j3 = j23/strides_[2]; 
    ptr_idx[0] = j0 + offsets_[0];
    ptr_idx[1] = j1 + offsets_[1];
    ptr_idx[2] = j2 + offsets_[2];
    ptr_idx[3] = j3 + offsets_[3];
  }

  // For LayoutRight
  template <size_t NDIM = ND, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutRight>>::value &&
                                 std::is_same<std::integral_constant<size_t, NDIM>, std::integral_constant<size_t, 1>>::value, void>::type
  to_coord(const int idx, int *ptr_idx) const noexcept {
    ptr_idx[0] = idx + offsets_[0];
  }

  template <size_t NDIM = ND, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutRight>>::value &&
                                 std::is_same<std::integral_constant<size_t, NDIM>, std::integral_constant<size_t, 2>>::value, void>::type
  to_coord(const int idx, int *ptr_idx) const noexcept {
    int j0 = idx/strides_[1], j1 = idx%strides_[1];
    ptr_idx[0] = j0 + offsets_[0];
    ptr_idx[1] = j1 + offsets_[1];
  }

  template <size_t NDIM = ND, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutRight>>::value &&
                                 std::is_same<std::integral_constant<size_t, NDIM>, std::integral_constant<size_t, 3>>::value, void>::type
  to_coord(const int idx, int *ptr_idx) const noexcept {
    int j01 = idx/strides_[2], j2 = idx%strides_[2];
    int j0  = j01/strides_[1], j1 = j01%strides_[1];
    ptr_idx[0] = j0 + offsets_[0];
    ptr_idx[1] = j1 + offsets_[1];
    ptr_idx[2] = j2 + offsets_[2];
  }

  template <size_t NDIM = ND, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutRight>>::value &&
                                 std::is_same<std::integral_constant<size_t, NDIM>, std::integral_constant<size_t, 4>>::value, void>::type
  to_coord(const int idx, int *ptr_idx) const noexcept {
    int j012 = idx/strides_[3], j3 = idx%strides_[3];
    int j01 = j012/strides_[2], j2 = j012%strides_[2];
    int j0  = j01/strides_[1], j1 = j01%strides_[1];
    ptr_idx[0] = j0 + offsets_[0];
    ptr_idx[1] = j1 + offsets_[1];
    ptr_idx[2] = j2 + offsets_[2];
    ptr_idx[3] = j3 + offsets_[3];
  }
};

/*
 * Helper class to merge multi-dimensional indices into flatten 1D index
 */
template <size_t ND, Layout LayoutType = Layout::LayoutLeft>
struct ViewIndex {
  int total_offset_;
  int strides_[ND] = {};
  size_t dims_ = ND;

  using layout_ = std::integral_constant<Layout, LayoutType>;

  ViewIndex() : total_offset_(0) {}

  // Constructor instanized with shape_nd
  ViewIndex(const shape_nd<ND>& strides)
    : total_offset_(0) {
    set(strides);
  }

  // Kokkos like constructor
  template <typename... I>
  ViewIndex(I... indices)
    : total_offset_(0) {
    set(indices...);
  }

  // Constructor with Offset
  ViewIndex(const shape_nd<ND>& strides, const shape_nd<ND>& offsets)
    : total_offset_(0) {
    set(strides, offsets);
  }

  ~ViewIndex() { }

  ViewIndex(const ViewIndex &rhs) {
    this->total_offset_ = rhs.total_offset_;
    for(int i=0; i<dims_; i++) {
      this->strides_[i] = rhs.strides(i);
    }
  }

  ViewIndex& operator=(const ViewIndex &rhs) {
    this->total_offset_ = rhs.total_offset_;
    for(int i=0; i<dims_; i++) {
      this->strides_[i] = rhs.strides(i);
    }

    return *this;
  }

  void set(const shape_nd<ND>& strides) {
    total_offset_ = 0;
    shape_nd<ND> offsets({0});
    init(strides, offsets);
  }

  template <typename... I>
  void set(I... indices) {
    static_assert(sizeof...(I) == ND, "The number of indices must be equal to ND");
    total_offset_ = 0;
    shape_nd<ND> offsets({0});
    shape_nd<ND> strides_meta;
    
    // expand parameter packs in the initializer
    int indices_tmp[ND] = {indices...};
    for(int i=0; i<ND; i++) {
      strides_meta[i] = indices_tmp[i];
    }
    init(strides_meta, offsets);
  }

  void set(const shape_nd<ND>& strides, const shape_nd<ND>& offsets) {
    total_offset_ = 0;
    init(strides, offsets);
  }

  void init(const shape_nd<ND>& strides_meta, const shape_nd<ND>& offsets_meta) {
    using layout_left = std::integral_constant<Layout, Layout::LayoutLeft>;
    if(std::is_same<layout_, layout_left>::value) {
      int total_strides = 1;
      for(int i=0; i<ND; i++) {
        total_strides *= strides_meta[i];
        strides_[i] = total_strides;
      }
    } else {
      int total_strides = 1;
      for(int i=0; i<ND; i++) {
        total_strides *= strides_meta[ND-1-i];
        strides_[ND-1-i] = total_strides;
      }
    }

    // compute the total offsets
    int offset = 0;
    if(std::is_same<layout_, layout_left>::value) {
      offset -= offsets_meta[0];
      int total_strides = 1;
      for(int i=0; i<ND-1; i++) {
        total_strides *= strides_meta[i];
        offset -= offsets_meta[i+1] * total_strides;
      }
    } else {
      offset -= offsets_meta[ND-1];
      int total_strides = 1;
      for(int i=1; i<ND; i++) {
        total_strides *= strides_meta[ND-i];
        offset -= offsets_meta[ND-1-i] * total_strides;
      }
    }
    total_offset_ = offset;
  }

  size_t dims() const {return dims_;}
  int* strides() const noexcept {return strides_;}
  int strides(const int i) const noexcept {return strides_[i];}
  Layout layout() const {
    using layout_left = std::integral_constant<Layout, Layout::LayoutLeft>;
    if(std::is_same<layout_, layout_left>::value) {
      return Layout::LayoutLeft;
    } else {
      return Layout::LayoutRight;
    }
  }
  inline int total_offsets() const noexcept {return total_offset_;}
  
  template <typename... I>
  inline int operator()(I... indices) const noexcept {
    static_assert(sizeof...(I) == ND, "The number of indices must be equal to ND");
    return flatten_(indices...);
  }

  template <typename... I>
  inline int operator[](I... indices) const noexcept {
    static_assert(sizeof...(I) == ND, "The number of indices must be equal to ND");
    return flatten_(indices...);
  }

private:
  // For LayoutLeft
  template <typename I0, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutLeft>>::value, int>::type
  flatten_(I0 i0) const noexcept {
    return total_offset_ + i0;
  }

  template <typename I0, typename I1, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutLeft>>::value, int>::type
  flatten_(I0 i0, I1 i1) const noexcept {
    int idx = total_offset_ + i0 + i1 * strides_[0];
    return idx;
  }

  template <typename I0, typename I1, typename I2, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutLeft>>::value, int>::type
  flatten_(I0 i0, I1 i1, I2 i2) const noexcept {
    int idx = total_offset_ + i0 + i1 * strides_[0] + i2 * strides_[1];
    return idx;
  }

  template <typename I0, typename I1, typename I2, typename I3, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutLeft>>::value, int>::type
  flatten_(I0 i0, I1 i1, I2 i2, I3 i3) const noexcept {
    int idx = total_offset_ + i0 + i1 * strides_[0] + i2 * strides_[1] + i3 * strides_[2];
    return idx;
  }

  // For LayoutRight
  template <typename I0, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutRight>>::value, int>::type
  flatten_(I0 i0) const noexcept {
    return total_offset_ + i0;
  }

  template <typename I0, typename I1, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutRight>>::value, int>::type
  flatten_(I0 i0, I1 i1) const noexcept {
    int idx = total_offset_ + i1 + i0 * strides_[1];
    return idx;
  }

  template <typename I0, typename I1, typename I2, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutRight>>::value, int>::type
  flatten_(I0 i0, I1 i1, I2 i2) const noexcept {
    int idx = total_offset_ + i2 + i1 * strides_[2] + i0 * strides_[1];
    return idx;
  }

  template <typename I0, typename I1, typename I2, typename I3, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutRight>>::value, int>::type
  flatten_(I0 i0, I1 i1, I2 i2, I3 i3) const noexcept {
    int idx = total_offset_ + i3 + i2 * strides_[3] + i1 * strides_[2] + i0 * strides_[1];
    return idx;
  }
};

template <typename ScalarType, size_t ND, Layout LayoutType = Layout::LayoutLeft>
class View {
  std::string name_;

  // flatten helper
  using IndexType = ViewIndex<ND, LayoutType>;
  IndexType index_;

  // In case instanized with default constructor
  bool is_empty_;

  // Raw data is kept by std::vector
  std::vector<ScalarType> data_;

  // Meta data used in offload region
  std::vector<int> strides_;
  int total_offset_;

  // Used outside offload region
  shape_nd<ND> strides_meta_;
  shape_nd<ND> offsets_meta_;
  shape_nd<ND> end_meta_;
  size_t size_; // total data size
  size_t dims_ = ND;

public:
  using layout_ = std::integral_constant<Layout, LayoutType>;
  using value_type_ = ScalarType;

public:
  // Default constructor, define an empty view
  View() : name_("empty"), total_offset_(0), size_(0),
           strides_meta_ {0}, offsets_meta_ {0}, end_meta_ {0}, is_empty_(true)
  {}

  // Constructor instanized with shape_nd
  View(const std::string name, const shape_nd<ND>& strides)
    : name_(name), total_offset_(0), strides_meta_(strides), is_empty_(false) {
    index_.set(strides);
    offsets_meta_.fill(0);
    init();
  }

  // Kokkos like constructor
  template <typename... I>
  View(const std::string name, I... indices)
    : name_(name), total_offset_(0), is_empty_(false) {
    static_assert(sizeof...(I) == ND, "The number of indices must be equal to ND");
    offsets_meta_.fill(0);

    // expand parameter packs in the initializer
    int indices_tmp[ND] = {indices...};
    for(int i=0; i<ND; i++) {
      strides_meta_[i] = indices_tmp[i];
    }
    index_.set(indices...);
    init();
  }

  // View with offsets
  View(const std::string name, const shape_nd<ND>& strides, const shape_nd<ND>& offsets)
    : name_(name), total_offset_(0), strides_meta_(strides), offsets_meta_(offsets), is_empty_(false) {
    index_.set(strides, offsets);
    init();
  }

  ~View() { free(); }

  void init() {
    strides_.resize(ND);

    for(int i=0; i<ND; i++) {
      end_meta_[i] = strides_meta_[i] + offsets_meta_[i];
    }

    using layout_left = std::integral_constant<Layout, Layout::LayoutLeft>;
    if(std::is_same<layout_, layout_left>::value) {
      int total_strides = 1;
      for(int i=0; i<ND; i++) {
        total_strides *= strides_meta_[i];
        strides_[i] = total_strides;
      }
    } else {
      int total_strides = 1;
      for(int i=0; i<ND; i++) {
        total_strides *= strides_meta_[ND-1-i];
        strides_[ND-1-i] = total_strides;
      }
    }

    int size = 1;
    for(auto&& dim : strides_meta_)
      size *= dim;
    size_ = size;
    data_.resize(size);

    // subtract the offsets here
    int offset = 0;
    if(std::is_same<layout_, layout_left>::value) {
      offset -= offsets_meta_[0];
      int total_strides = 1;
      for(int i=0; i<ND-1; i++) {
        total_strides *= strides_meta_[i];
        offset -= offsets_meta_[i+1] * total_strides;
      }
    } else {
      offset -= offsets_meta_[ND-1];
      int total_strides = 1;
      for(int i=1; i<ND; i++) {
        total_strides *= strides_meta_[ND-i];
        offset -= offsets_meta_[ND-1-i] * total_strides;
      }
    }
    total_offset_ = offset;
  }

  void free() {
    if(!is_empty_) {
      std::vector<ScalarType>().swap(data_);
      std::vector<int>().swap(strides_);
    }
  }

  // Copy constructor performs the DEEP copy
  View(const View &rhs) {
    copy(rhs); 
    setName(name_ + "_copy");
  }

  // Assignment operator used only for data allocation, performs the DEEP copy
  View& operator=(const View &rhs) {
    copy(rhs); 
    setName(name_ + "_assign");
    //initiate();
    return *this; 
  }

  template <typename... I>
  inline ScalarType& operator()(I... indices) noexcept {
    static_assert(sizeof...(I) == ND, "The number of indices must be equal to ND");
    return access(indices...);
  }

  template <typename... I>
  inline ScalarType operator()(I... indices) const noexcept {
    static_assert(sizeof...(I) == ND, "The number of indices must be equal to ND");
    return access(indices...);
  }

  template <typename... I>
  inline ScalarType& operator[](I... indices) noexcept {
    static_assert(sizeof...(I) == ND, "The number of indices must be equal to ND");
    return access(indices...);
  }

  template <typename... I>
  inline ScalarType operator[](I... indices) const noexcept {
    static_assert(sizeof...(I) == ND, "The number of indices must be equal to ND");
    return access(indices...);
  }

  // Do nothing, in order to copy the data to host
  void updateSelf() {
    auto tmp = data_;
    std::copy(tmp.begin(), tmp.end(), data_.begin());
  }

  // Do nothing, in order to copy the data to device
  void updateDevice() {
    auto tmp = data_;
    std::copy(std::execution::par_unseq, tmp.begin(), tmp.end(), data_.begin());
  }

  // Do nothing, run GPU kernel to move data to device
  void initiate() {
    std::vector<ScalarType> tmp(1);
    std::copy(std::execution::par_unseq, tmp.begin(), tmp.end(), data_.begin());
  }

  // Deep copy the rhs instance
  void copy(const View &rhs) {
    this->free();

    data_ = rhs.vector();
    strides_ = rhs.strides();
    index_ = rhs.index();

    this->setIsEmpty(rhs.is_empty());
    this->setSize(rhs.size());
    this->setDims(rhs.dims());
    this->setStridesMeta(rhs.strides_meta());
    this->setOffsetsMeta(rhs.offsets()); 
    this->setTotalOffsets(rhs.total_offsets());
    this->setEndMeta(rhs.end());
    this->setName(rhs.name());
  }

  void fill(const ScalarType value = 0) {
    std::fill(data_.begin(), data_.end(), value);
  }

  void swap(View &rhs) {
    // Temporal data
    bool is_empty        = this->is_empty_;
    std::string name     = this->name_;
    int total_offset     = this->total_offset_;
    shape_nd<ND> strides_meta = this->strides_meta_;
    shape_nd<ND> offsets_meta = this->offsets_meta_;
    shape_nd<ND> end_meta     = this->end_meta_;
    size_t size = this->size_;
    size_t dims = this->dims_;

    // Update this
    this->setIsEmpty(rhs.is_empty());
    this->setSize(rhs.size());
    this->setDims(rhs.dims());
    this->setStridesMeta(rhs.strides_meta());
    this->setOffsetsMeta(rhs.offsets()); 
    this->setTotalOffsets(rhs.total_offsets());
    this->setEndMeta(rhs.end());
    this->setName(rhs.name());

    // Update the rhs
    rhs.setIsEmpty(is_empty);
    rhs.setSize(size);
    rhs.setDims(dims);
    rhs.setStridesMeta(strides_meta);
    rhs.setOffsetsMeta(offsets_meta);
    rhs.setTotalOffsets(total_offset);
    rhs.setEndMeta(end_meta);
    rhs.setName(name);

    // Swap the data and strides
    data_.swap(rhs.vector());
    strides_.swap(rhs.strides());
  }

public:
  // Getters
  IndexType index() const { return index_; }
  IndexType &index() { return index_; }
  bool is_empty() const { return is_empty_; }
  size_t size() const { return size_; }
  size_t dims() const { return dims_; }

  std::vector<ScalarType> vector() const { return data_; }
  std::vector<ScalarType> &vector() { return data_; }
  ScalarType *data() { return data_.data(); }
  const ScalarType *data() const { return data_.data(); }
  std::vector<int> strides() const { return strides_; }
  std::vector<int> &strides() { return strides_; }

  inline const shape_nd<ND>& strides_meta() const noexcept { return strides_meta_; }
  inline const shape_nd<ND>& offsets() const noexcept { return offsets_meta_; }
  inline const shape_nd<ND>& begin() const noexcept { return offsets_meta_; }
  inline const shape_nd<ND>& end() const noexcept { return end_meta_; }
  inline int strides(size_t i) const noexcept { return strides_meta_[i]; }
  inline int offsets(size_t i) const noexcept {return offsets_meta_[i];}
  inline int begin(size_t i) const noexcept {return offsets_meta_[i];}
  inline int end(size_t i) const noexcept {return end_meta_[i];}
  inline int total_offsets() const noexcept {return total_offset_;}
  std::string name() const noexcept {return name_;}

  // Setters
  inline void setIsEmpty(bool is_empty) { is_empty_ = is_empty; }
  inline void setSize(size_t size) { size_ = size; }
  inline void setDims(size_t dims) { dims_ = dims; }
  inline void setStrides(const std::vector<int> &strides) {
    strides_ = strides;
  }
  inline void setStridesMeta(const shape_nd<ND>& strides_meta) {
    strides_meta_ = strides_meta;
  }
  inline void setOffsetsMeta(const shape_nd<ND>& offsets_meta) {
    offsets_meta_ = offsets_meta;
  }
  inline void setEndMeta(const shape_nd<ND>& end_meta) {
    end_meta_ = end_meta;
  }
  inline void setTotalOffsets(const int total_offset) {
    total_offset_ = total_offset;
  }
  inline void setName(std::string name) {
    name_ = name;
  }

private:
  // Naive accessors to ease the compiler optimizations
  // For LayoutLeft
  template <typename I0, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutLeft>>::value, ScalarType&>::type
  access(I0 i0) noexcept {
    return data_[total_offset_ + i0];
  }

  template <typename I0, typename I1, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutLeft>>::value, ScalarType&>::type
  access(I0 i0, I1 i1) noexcept {
    int idx = total_offset_ + i0 + i1 * strides_[0];
    return data_[idx];
  }

  template <typename I0, typename I1, typename I2, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutLeft>>::value, ScalarType&>::type
  access(I0 i0, I1 i1, I2 i2) noexcept {
    int idx = total_offset_ + i0 + i1 * strides_[0] + i2 * strides_[1];
    return data_[idx];
  }

  template <typename I0, typename I1, typename I2, typename I3, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutLeft>>::value, ScalarType&>::type
  access(I0 i0, I1 i1, I2 i2, I3 i3) noexcept {
    int idx = total_offset_ + i0 + i1 * strides_[0] + i2 * strides_[1] + i3 * strides_[2];
    return data_[idx];
  }

  template <typename I0, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutLeft>>::value, ScalarType>::type
  access(I0 i0) const noexcept {
    return data_[total_offset_ + i0];
  }

  template <typename I0, typename I1, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutLeft>>::value, ScalarType>::type
  access(I0 i0, I1 i1) const noexcept {
    int idx = total_offset_ + i0 + i1 * strides_[0];
    return data_[idx];
  }

  template <typename I0, typename I1, typename I2, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutLeft>>::value, ScalarType>::type
  access(I0 i0, I1 i1, I2 i2) const noexcept {
    int idx = total_offset_ + i0 + i1 * strides_[0] + i2 * strides_[1];
    return data_[idx];
  }

  template <typename I0, typename I1, typename I2, typename I3, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutLeft>>::value, ScalarType>::type
  access(I0 i0, I1 i1, I2 i2, I3 i3) const noexcept {
    int idx = total_offset_ + i0 + i1 * strides_[0] + i2 * strides_[1] + i3 * strides_[2];
    return data_[idx];
  }

  // For LayoutRight
  template <typename I0, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutRight>>::value, ScalarType&>::type
  access(I0 i0) noexcept {
    return data_[total_offset_ + i0];
  }

  template <typename I0, typename I1, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutRight>>::value, ScalarType&>::type
  access(I0 i0, I1 i1) noexcept {
    int idx = total_offset_ + i1 + i0 * strides_[1];
    return data_[idx];
  }

  template <typename I0, typename I1, typename I2, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutRight>>::value, ScalarType&>::type
  access(I0 i0, I1 i1, I2 i2) noexcept {
    int idx = total_offset_ + i2 + i1 * strides_[2] + i0 * strides_[1];
    return data_[idx];
  }

  template <typename I0, typename I1, typename I2, typename I3, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutRight>>::value, ScalarType&>::type
  access(I0 i0, I1 i1, I2 i2, I3 i3) noexcept {
    int idx = total_offset_ + i3 + i2 * strides_[3] + i1 * strides_[2] + i0 * strides_[1];
    return data_[idx];
  }

  template <typename I0, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutRight>>::value, ScalarType>::type
  access(I0 i0) const noexcept {
    return data_[total_offset_ + i0];
  }

  template <typename I0, typename I1, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutRight>>::value, ScalarType>::type
  access(I0 i0, I1 i1) const noexcept {
    int idx = total_offset_ + i1 + i0 * strides_[1];
    return data_[idx];
  }

  template <typename I0, typename I1, typename I2, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutRight>>::value, ScalarType>::type
  access(I0 i0, I1 i1, I2 i2) const noexcept {
    int idx = total_offset_ + i2 + i1 * strides_[2] + i0 * strides_[1];
    return data_[idx];
  }

  template <typename I0, typename I1, typename I2, typename I3, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutRight>>::value, ScalarType>::type
  access(I0 i0, I1 i1, I2 i2, I3 i3) const noexcept {
    int idx = total_offset_ + i3 + i2 * strides_[3] + i1 * strides_[2] + i0 * strides_[1];
    return data_[idx];
  }
};

#endif
