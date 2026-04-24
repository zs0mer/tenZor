#pragma once

#include <array>
#include <cstdint>

#include "allocator.hpp"
#include "buffer.hpp"
#include "kernel_functions.hpp"

namespace TZ::internal {

template <class T>
// standard tensor class
class TensorIMPL {
  protected:
	uint8_t dim_;
	std::array<uint64_t, MAX_DIM> shape_;
	std::array<uint64_t, MAX_DIM> strides_;
	uint64_t offset_;
	mem::Buffer data_;

	uint64_t size_;
	bool dense_;

	// ! The data may not be layed linearly in memory

	// * the tensor can be:
	// * - Normal - dimension: anything   - shape: anything
	// * - Scalar - dimension: 0          - shape: {}
	// * - Empty  - dimension: 0          - shape: has at least one 0 in it
	// * - Null   - dimension: 0          - shape: {}

	// * normal index: array of the indexes to each dimension
	// * linear index: the way to index the memory, it only works with rawData()

	// ! you can bypass the const correctness with the buffer()
	// ! or with the copy constructor and the move constructor

  public:
	// # seters ===========================================================================

	TensorIMPL();

	// standard constructor
	// the first argument is the number of dimensions the tensor has
	// the second argument is a pointer to a C style array containing the shape of the tensor
	TensorIMPL(const uint8_t dim, const uint64_t* shape, Device device);

	// standard constructor
	TensorIMPL(const std::initializer_list<uint64_t>& shape, Device device);

	// constructor, should only use it with caution
	TensorIMPL(const uint8_t dim, const uint64_t* shape, const uint64_t* strides,
	           const uint64_t offset, mem::Buffer data);

	// makes a new Tensor
	void set(const std::initializer_list<uint64_t>& shape, Device device);

	// makes a new Tensor
	void set(const uint64_t dim, const uint64_t* shape, Device device);

	// makes a new Tensor
	void set(const uint8_t dim, const uint64_t* shape, const uint64_t* strides,
	         const uint64_t offset, mem::Buffer data);


	TensorIMPL(const TensorIMPL&) = default;

	// * if TZ_NORMAL_EQUAL is 1:
	// * if the tensors have the same shape and are on the same device,
	// * it will just copy the data
	TensorIMPL<T>& operator=(const TensorIMPL<T>&);

	TensorIMPL(TensorIMPL&&) = default;

	// * if TZ_NORMAL_EQUAL is 1:
	// * if the tensors have the same shape and are on the same device,
	// * it will just copy the data
	TensorIMPL<T>& operator=(TensorIMPL<T>&&);

	// # metadata geters ==================================================================

	// get the dimension of the tensor
	uint64_t dim() const;

	// get the shape of the tensor
	const uint64_t* shape() const;

	// get the strides of the tensor
	const uint64_t* strides() const;

	// get the number of elements in the tensor
	uint64_t size() const;

	// pointer to the start of the data buffer to this tensor
	T* data();

	// pointer to the start of the data buffer to this tensor
	const T* data() const;

	// get the pointer to the start of the whole tensor buffer
	// this will not necessarily start where the data is located
	T* rawData();

	// get the pointer to the start of the whole tensor buffer
	// this will not necessarily start where the data is located
	const T* rawData() const;

	// returns the offset of the first element of the tensor in the buffer
	uint64_t offset() const;

	// the tensor is layed out flat in memory
	bool dense() const;

	// returns true if the tensor is a scalar
	bool scalar() const;

	// returns true if the tensor is Empty or Null
	// returns size() == 0
	bool empty() const;

	// returns true if the tensor is normal
	bool indexable() const;

	// returns the allocator, what allocated this buffer
	mem::Allocator& allocator() const;

	// returns the device the memory is on
	Device device() const;

	// returns the TZ::mem::Buffer object, that has the memory
	mem::Buffer buffer();

	// returns the TZ::mem::Buffer object, that has the memory
	// ! with this method you modify the data for a const Tensor
	const mem::Buffer buffer() const;

	// # geters ===========================================================================

	// returns the data containing in the given index
	// cant do on the GPU
	T& at(const std::vector<uint64_t>& idx);

	// returns the data containing in the given index
	// cant do on the GPU
	const T& at(const std::vector<uint64_t>& idx) const;

	// returns the data containing in the given index
	// the input is a C style array containing dim number of indexes
	// cant do on the GPU
	T& at(const uint64_t* idx);

	// returns the data containing in the given index
	// the input is a C style array containing dim number of indexes
	// cant do on the GPU
	const T& at(const uint64_t* idx) const;

	// returns a Tensor containing the data in the given index
	// its just a view
	// if the remaining tensor is a scalar then it will return a scalar Tensor
	TensorIMPL<T> operator[](const uint64_t idx);

	// returns a Tensor containing the data in the given index
	// its just a view
	// if the remaining tensor is a scalar then it will return a scalar Tensor
	const TensorIMPL<T> operator[](const uint64_t idx) const;

	// makes a new tensor that has the same data as the old one
	TensorIMPL<T> clone() const;

	// makes a new tensor that has the same data as the old one
	// can be on a new device
	TensorIMPL<T> copyTo(Device device) const;

	// IF the tensor is scalar it returns the one element
	// cant do on the GPU
	T& get();

	// IF the tensor is scalar it returns the one element
	// cant do on the GPU
	const T& get() const;

	// # operators ========================================================================

	// * static
	// calls func(a[i]) for all elements of the tensor
	// you can modify a
	template <class Func>
	static void apply(TensorIMPL<T>& a, Func func);

	// * static
	// calls func(a[i], b[i]) for all elements of the tensor
	// you can modify b
	template <class Func>
	static void apply(const TensorIMPL<T>& a, TensorIMPL<T>& b, Func func);

	// * static
	// calls func(a[i], b[i], c[i]) for all elements of the tensor
	// you can modify c
	template <class Func>
	static void apply(const TensorIMPL<T>& a, const TensorIMPL<T>& b, TensorIMPL<T>& c, Func func);


	// calls func(this[i]) for all elements of the tensor
	// you can modify "this[i]"
	template <class Func>
	void apply(Func func);

	// calls func(a[i], this[i]) for all elements of the tensor
	// you can modify "this[i]"
	template <class Func>
	void apply(const TensorIMPL<T>& a, Func func);

	// calls func(a[i], b[i], this[i]) for all elements of the tensor
	// you can modify "this[i]"
	template <class Func>
	void apply(const TensorIMPL<T>& a, const TensorIMPL<T>& b, Func func);


	cuda::SimpleTensor<T> getCudaTensor();

	const cuda::SimpleTensor<T> getCudaTensor() const;

	// # private ==========================================================================
  protected:
	// # helper functions ---------------------

	void computeStrides();

	void computeMetadata();

	static bool isSameShape(const TensorIMPL<T>& a, const TensorIMPL<T>& b);
};

} // namespace TZ::internal
