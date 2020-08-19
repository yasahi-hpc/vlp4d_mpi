/*
 *  Simplified View to hand multidimensional array inside OpenACC accelerated region
 *
 */

#ifndef __OPENACC_VIEW_HPP__
#define __OPENACC_VIEW_HPP__

#include <iostream>
#include <string>
#include <type_traits>
#include <array>

#if defined( ENABLE_OPENACC )
  #include <openacc.h>
#endif

#include "ViewLayout.hpp"

template <size_t ND>
using shape_nd = std::array<int, ND>;

template <typename ScalarType, size_t ND, Layout LayoutType = Layout::LayoutLeft>
class View {
  std::string name_;

  // Do not delete the pointers, if this instance is a copy
  bool is_copied_;

  // In case instanized with default constructor
  bool is_empty_;

  // Pointer with offsets (just a reference to data_raw)
  ScalarType *data_; 

  // Pointer without offset
  ScalarType *data_raw_;

  // Used in offload region
  int *strides_;
  int total_offset_;

  // Used outside offload region
  shape_nd<ND> strides_meta_;
  shape_nd<ND> offsets_meta_;
  size_t size_; // total data size
  size_t dims_ = ND;

public:
  typedef std::integral_constant<Layout, LayoutType> layout_;
  typedef ScalarType value_type_;

public:
  // Default constructor, define an empty view
  View() : total_offset_(0), size_(0), data_(nullptr), data_raw_(nullptr),
           strides_meta_ {0}, offsets_meta_ {0}, is_copied_(false), is_empty_(true) {
    name_ = "empty";
  }

  View(const std::string name, const shape_nd<ND>& strides)
    : total_offset_(0), name_(name), strides_meta_(strides), is_copied_(false), is_empty_(false) {
    offsets_meta_.fill(0);

    strides_ = new int[ND];
    typedef std::integral_constant<Layout, Layout::LayoutLeft> layout_left;
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

    size_t sz = 1;
    for(auto&& dim : strides_meta_)
      sz *= dim;
    data_raw_ = new ScalarType[sz];
    data_ = data_raw_;
    size_ = sz;

    for(int i=0; i<sz; i++) {
      this->data_raw_[i] = 0.;
    }

    #if defined( ENABLE_OPENACC )
      #pragma acc enter data copyin(this) // shallow copy this pointer
      #pragma acc enter data create(data_[0:size_], strides_[0:dims_]) // attach data (deep copy)
      #pragma acc update device(data_[0:size_], strides_[0:dims_])
    #endif
  }

  // Kokkos like constructor
  template <typename... I>
  View(const std::string name, I... indices)
    : total_offset_(0), name_(name), is_copied_(false), is_empty_(false) {
    static_assert(sizeof...(I) == ND, "The number of indices must be equal to ND");
    offsets_meta_.fill(0);
   
    // expand parameter packs in the initializer
    int indices_tmp[ND] = {indices...};

    strides_ = new int[ND];
   
    for(int i=0; i<ND; i++) {
      strides_meta_[i] = indices_tmp[i];
    }

    typedef std::integral_constant<Layout, Layout::LayoutLeft> layout_left;
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
   
    size_t sz = 1;
    for(auto&& dim : strides_meta_)
      sz *= dim;
    data_raw_ = new ScalarType[sz];
    data_ = data_raw_;
    size_ = sz;
   
    for(int i=0; i<sz; i++) {
      this->data_raw_[i] = 0.;
    }
   
    #if defined( ENABLE_OPENACC )
      #pragma acc enter data copyin(this) // shallow copy this pointer
      #pragma acc enter data create(data_[0:size_], strides_[0:dims_]) // attach data (deep copy)
      #pragma acc update device(data_[0:size_], strides_[0:dims_])
    #endif
  }
  

  View(const std::string name, const shape_nd<ND>& strides, const shape_nd<ND>& offsets)
    : total_offset_(0), name_(name), strides_meta_(strides), offsets_meta_(offsets), is_copied_(false), is_empty_(false) {

    strides_ = new int[ND];

    typedef std::integral_constant<Layout, Layout::LayoutLeft> layout_left;
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

    size_t sz = 1;
    for(auto&& dim : strides_meta_)
      sz *= dim;
    size_ = sz;
    data_raw_ = new ScalarType[sz];
    data_     = data_raw_;

    for(int i=0; i<sz; i++) {
      this->data_raw_[i] = 0.;
    }

    // subtract the offsets here
    int offset = 0;
    typedef std::integral_constant<Layout, Layout::LayoutLeft> layout_left;
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
    data_ = data_raw_ + offset;

    #if defined( ENABLE_OPENACC )
      #pragma acc enter data copyin(this) // shallow copy this pointer
      #pragma acc enter data create(data_[0:size_], strides_[0:dims_]) // attach data
      #pragma acc update device(data_[0:size_], strides_[0:dims_])
    #endif
  }

  ~View() {
    if(! is_empty_) {
      #if defined( ENABLE_OPENACC )
        // In case, this is not a copy, deallocate the data
        if(!is_copied_) {
          #pragma acc exit data delete(data_[0:size_], strides_[0:dims_]) // detach data
          if(data_raw_ != nullptr) delete [] data_raw_;
          if(strides_  != nullptr) delete [] strides_;

          data_raw_ = nullptr;
          strides_  = nullptr;
        }
        #pragma acc exit data delete(this) // delete this pointer
      #endif
    }
  }

public:
  // Copy and assignment, implemented as deep copy
  // Overwrite the original view, used for initialization
  View(const View &rhs)
    : total_offset_(rhs.total_offsets()), strides_meta_(rhs.strides()), offsets_meta_(rhs.offsets()),
    is_copied_(false), is_empty_(false) {
    // First check if the layout and datatype is identical
    this->name_     = rhs.name() + "_copy";
    this->strides_  = new int[rhs.dims()];
    this->data_raw_ = new ScalarType[rhs.size()];
    this->size_     = rhs.size();
    this->dims_     = rhs.dims();
    
    // copy meta data
    for(int i=0; i<ND; i++) {
      this->strides_[i] = rhs.strides_ptr()[i];
    }
    
    size_t sz = this->size_;
    for(int i=0; i<sz; i++) {
      this->data_raw_[i] = rhs.data()[i];
    }
    this->data_ = this->data_raw_ + this->total_offset_;
    
    #if defined( ENABLE_OPENACC )
      #pragma acc enter data copyin(this) // shallow copy this pointer
      #pragma acc enter data create(data_[0:size_], strides_[0:dims_]) // attach data (deep copy)
      #pragma acc update device(data_[0:size_], strides_[0:dims_])
    #endif
  }

  View& operator=(const View &rhs) {
    this->is_empty_  = false;
    this->is_copied_ = false;
    this->strides_  = new int[rhs.dims()];
    this->data_raw_ = new ScalarType[rhs.size()];
    this->size_     = rhs.size();
    this->dims_     = rhs.dims();
    this->name_     = rhs.name() + "_copy";
    this->total_offset_ = rhs.total_offsets();
    
    // copy meta data
    for(int i=0; i<ND; i++) {
      this->strides_meta_[i] = rhs.strides()[i];
      this->offsets_meta_[i] = rhs.offsets()[i];
      this->strides_[i]      = rhs.strides_ptr()[i];
    }
    
    size_t sz = this->size_;
    for(int i=0; i<sz; i++) {
      this->data_raw_[i] = rhs.data()[i];
    }
    this->data_ = this->data_raw_ + this->total_offset_;
    
    #if defined( ENABLE_OPENACC )
      #pragma acc enter data copyin(this) // shallow copy this pointer
      #pragma acc enter data create(data_[0:size_], strides_[0:dims_]) // attach data (deep copy)
      #pragma acc update device(data_[0:size_], strides_[0:dims_])
    #endif
    
    return *this;
  }

public:
  template <typename... I>
  inline ScalarType& operator()(I... indices) const noexcept {
    static_assert(sizeof...(I) == ND, "The number of indices must be equal to ND");
    return access(indices...);
  }

  template <typename... I>
  inline ScalarType& operator[](I... indices) const noexcept {
    static_assert(sizeof...(I) == ND, "The number of indices must be equal to ND");
    return access(indices...);
  }

public:
  // methods for device/host data transfer
  inline void updateSelf() {
    #if defined( ENABLE_OPENACC )
      #pragma acc update self(data_[0:size_])
    #endif
  }

  inline void updateDevice() {
    #if defined( ENABLE_OPENACC )
      #pragma acc update device(data_[0:size_])
    #endif
  }

  // Shallow copy the rhs value.
  // If destructor called, delete the reference only
  inline void copy(View &rhs) {
    this->is_copied_ = true;
    setSize(rhs.size());
    setDims(rhs.dims());
    setDataRaw(rhs.data());
    setData(rhs.data_ptr()); // attach the data pointer
    setStridesMeta(rhs.strides());
    setStrides(rhs.strides_ptr()); // attach the strides pointer
    setOffsetsMeta(rhs.offsets());
    setTotalOffsets(rhs.total_offsets());
    setName(rhs.name());
  }

  // Exchange meta data with another view
  void swap(View &rhs) {
    // temporal data
    bool is_copied       = this->is_copied_;
    std::string name     = this->name_;
    int *strides         = this->strides_;
    ScalarType *data     = this->data_;
    ScalarType *data_raw = this->data_raw_;
    int total_offset     = this->total_offset_;
    shape_nd<ND> strides_meta = this->strides_meta_;
    shape_nd<ND> offsets_meta = this->offsets_meta_;
    size_t size = this->size_;
    size_t dims = this->dims_;

    // Update this
    this->setIsCopied(rhs.is_copied());
    this->setSize(rhs.size());
    this->setDims(rhs.dims());
    this->setDataRaw(rhs.data());
    this->setData(rhs.data_ptr()); // attach the data pointer
    this->setStridesMeta(rhs.strides());
    this->setStrides(rhs.strides_ptr()); // attach the strides pointer
    this->setOffsetsMeta(rhs.offsets());
    this->setTotalOffsets(rhs.total_offsets());
    this->setName(rhs.name());

    // Update the rhs
    rhs.setIsCopied(is_copied);
    rhs.setSize(size);
    rhs.setDims(dims);
    rhs.setDataRaw(data_raw);
    rhs.setData(data); // attach the data pointer
    rhs.setStridesMeta(strides_meta);
    rhs.setStrides(strides); // attach the strides pointer
    rhs.setOffsetsMeta(offsets_meta);
    rhs.setTotalOffsets(total_offset);
    rhs.setName(name);
  }

public:
  // Getters
  bool is_copied() const {return is_copied_;}
  size_t size() const {return size_;}
  size_t dims() const {return dims_;}
  ScalarType *data() const {return data_raw_;}
  ScalarType* data_ptr() const {return data_;}

  Layout layout() const {
    typedef std::integral_constant<Layout, Layout::LayoutLeft> layout_left;
    if(std::is_same<layout_, layout_left>::value) {
      return Layout::LayoutLeft; 
    } else {
      return Layout::LayoutRight;
    }
  }

  inline const shape_nd<ND>& strides() const noexcept {return strides_meta_;}
  inline const shape_nd<ND>& offsets() const noexcept {return offsets_meta_;}
  inline int* strides_ptr() const noexcept {return strides_;}
  inline int strides(size_t i) const noexcept {return strides_meta_[i];}
  inline int offsets(size_t i) const noexcept {return offsets_meta_[i];}
  inline int total_offsets() const noexcept {return total_offset_;}
  std::string name() const noexcept {return name_;}

  // Setters
  inline void setIsCopied(bool is_copied) {is_copied_ = is_copied;}
  inline void setSize(size_t size) {size_ = size;}
  inline void setDims(size_t dims) {dims_ = dims;}
  inline void setDataRaw(ScalarType *data_raw) {
    data_raw_ = data_raw;
  }
  inline void setData(ScalarType *data) {
    data_ = data;
    #if defined( ENABLE_OPENACC )
      acc_attach( (void**) &data_ );
    #endif
  }
  inline void setStridesMeta(const shape_nd<ND>& strides_meta) {
    strides_meta_ = strides_meta;
  }
  inline void setOffsetsMeta(const shape_nd<ND>& offsets_meta) {
    offsets_meta_ = offsets_meta;
  }
  inline void setStrides(int *strides) {
    strides_ = strides;
    #if defined( ENABLE_OPENACC )
      acc_attach( (void**) &strides_ );
    #endif
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
  access(I0 i0) const noexcept {
    return data_[i0];
  }

  template <typename I0, typename I1, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutLeft>>::value, ScalarType&>::type 
  access(I0 i0, I1 i1) const noexcept {
    int idx = i0 + i1 * strides_[0];
    return data_[idx];
  }

  template <typename I0, typename I1, typename I2, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutLeft>>::value, ScalarType&>::type 
  access(I0 i0, I1 i1, I2 i2) const noexcept {
    int idx = i0 + i1 * strides_[0] + i2 * strides_[1];
    return data_[idx];
  }

  template <typename I0, typename I1, typename I2, typename I3, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutLeft>>::value, ScalarType&>::type 
  access(I0 i0, I1 i1, I2 i2, I3 i3) const noexcept {
    int idx = i0 + i1 * strides_[0] + i2 * strides_[1] + i3 * strides_[2];
    return data_[idx];
  }

  // For LayoutRight
  template <typename I0, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutRight>>::value, ScalarType&>::type 
  access(I0 i0) const noexcept {
    return data_[i0];
  }

  template <typename I0, typename I1, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutRight>>::value, ScalarType&>::type 
  access(I0 i0, I1 i1) const noexcept {
    int idx = i1 + i0 * strides_[1];
    return data_[idx];
  }

  template <typename I0, typename I1, typename I2, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutRight>>::value, ScalarType&>::type 
  access(I0 i0, I1 i1, I2 i2) const noexcept {
    int idx = i2 + i1 * strides_[2] + i0 * strides_[1];
    return data_[idx];
  }

  template <typename I0, typename I1, typename I2, typename I3, Layout LType = LayoutType>
  inline typename std::enable_if<std::is_same<std::integral_constant<Layout, LType>, std::integral_constant<Layout, Layout::LayoutRight>>::value, ScalarType&>::type 
  access(I0 i0, I1 i1, I2 i2) const noexcept {
    int idx = i3 + i2 * strides_[3] + i1 * strides_[2] + i0 * strides_[1];
    return data_[idx];
  }
};

#endif
