#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "allocator.hpp"
#include "buffer.hpp"

namespace TZ::internal {

// # -------------------------
#ifdef TZ_MAX_DIM
const constexpr std::uint8_t MAX_DIM = TZ_MAX_DIM;
#else
const constexpr std::uint8_t MAX_DIM = 64;
#endif
// # -------------------------

template <class T>
// standard tensor class
class TensorIMPL {
  protected:
	uint8_t dim_;
	std::array<uint64_t, MAX_DIM> shape_;
	std::array<uint64_t, MAX_DIM> strides_;
	uint64_t offset_;
	mem::Buffer data_;

	// ! The data may not be layed linearly in memory

	// * the tensor can be:
	// * - Normal - dimenson: anything   - shape: anything
	// * - Scalar - dimenson: 0          - shape: {}
	// * - Empty  - dimenson: 0          - shape: has at least one 0 in it
	// * - Null   - dimenson: 0          - shape: {}

	// * normal index: array of the indexes to each dimenson
	// * linear index: the way to index the memory, it only works with rawData()

  public:
	// # seters ===========================================================================

	TensorIMPL();

	// standard constructor
	// the first argument is the number of dimensons the tensor has
	// the second argument is a pointer to a C style array containing the shape of the tensor
	TensorIMPL(const uint8_t dim, const uint64_t* shape, Device device = CPU);

	// standard constructor
	TensorIMPL(const std::vector<uint64_t>& shape, Device device = CPU);

	// constructor, should only use it caution
	TensorIMPL(const uint8_t dim, const uint64_t* shape, const uint64_t* strides,
	           const uint64_t offset, mem::Buffer data);

	// makes a new Tensor
	void set(const std::vector<uint64_t>& shape, Device device = CPU);

	// makes a new Tensor
	void set(const uint64_t dim, const uint64_t* shape, Device device = CPU);


	TensorIMPL(const TensorIMPL&) = default;

	TensorIMPL<T>& operator=(const TensorIMPL<T>&);

	TensorIMPL(TensorIMPL&&) = default;

	TensorIMPL<T>& operator=(TensorIMPL<T>&&);

	// # metadata geters ==================================================================

	// get the dimenson of the tensor
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
	// if the remaining tensor is a scalar then it will return a saclar Tensor
	TensorIMPL<T> operator[](const uint64_t idx);

	// returns a Tensor containing the data in the given index
	// its just a view
	// if the remaining tensor is a scalar then it will return a saclar Tensor
	const TensorIMPL<T> operator[](const uint64_t idx) const;

	// makes a new tensor that has the same data as the old one
	TensorIMPL<T> clone() const;

	// IF the tensor is scalar
	// returns that one element
	// cant do on the GPU
	T& get();

	// IF the tensor is scalar
	// returns that one element
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

	// # private ==========================================================================
  protected:
	// # metadata cache

	template <class K>
	struct Cache {
		bool cached = false;
		K value;

		void set(const K& data) {
			value = data;
			cached = true;
		}

		void reset() {
			cached = false;
		}
	};

	mutable Cache<uint64_t> c_size_;
	mutable Cache<bool> c_dense;

	// # helper functions

	void computeStrides();

	static bool isSameShape(const TensorIMPL<T>& a, const TensorIMPL<T>& b);
};

} // namespace TZ::internal
