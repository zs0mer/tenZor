#pragma once
// only the declarations are in this file


namespace TZ {

template <class T>
// standard tensor class
class Tensor {
  private:
	std::vector<uint64_t> shape_;
	std::vector<uint64_t> strides_;
	uint64_t offset_;
	mem::Buffer data_;
	// cant have nullptr as data ptr
	// with a shape containing 0 its automaticly a scalar
	// the default tensor will allways be a scalar

  public:
	//& seters ===========================================================================

	Tensor();

	// standard constructor
	Tensor(const std::vector<uint64_t>& shape, const uint8_t alignment = DEFAULT_ALIGNMENT,
	       mem::Allocator& allocator = mem::salloc::instance());

	// un-nests a nested std::vector to a Tensor
	// has to be right shape
	template <class NestedVector>
	Tensor(const std::vector<NestedVector>& v, const uint8_t alignment = DEFAULT_ALIGNMENT,
	       mem::Allocator& allocator = mem::salloc::instance());

	// makes a new Tensor
	void set(const std::vector<uint64_t>& shape, const uint8_t alignment = DEFAULT_ALIGNMENT,
	         mem::Allocator& allocator = mem::salloc::instance());


	Tensor(const Tensor&) = default;

	Tensor& operator=(const Tensor&) = default;

	Tensor(Tensor&&) = default;

	Tensor& operator=(Tensor&&) = default;

	//& metadata geters ==================================================================

	// get the dimenson of the tensor
	uint64_t dim() const;

	// get the shape of the tensor in a vector
	const std::vector<uint64_t>& shape() const;

	// get the number of elements in the tensor
	uint64_t numel() const;

	// get the pointer to the start of the tensor buffer
	// not necessarily JUST the data for THIS tensor
	T* data(const bool fullBuffer = true);

	// get the pointer to the start of the tensor buffer
	// not necessarily JUST the data for this tensor
	const T* data(const bool fullBuffer = false) const;

	// returns true if this is not only a part of a bigger tensor
	// it means that the tensor is layed out flat in memory
	// if softCheck = true it can return true even if
	// the data is not at the start of the memory pointer, it is offseted by offset()
	bool isContiguous(const bool softCheck = false) const;

	// returns true if the tensor has 0 dimensons, and 1 element
	bool isScalar() const;

	//& geters ===========================================================================

	// returns a Tensor containing the data in the given index
	// its just a view
	// if the remaining tensor is a scalar then it will return a saclar Tensor
	T& at(const std::vector<uint64_t>& idx);

	// returns a Tensor containing the data in the given index
	// its just a view
	// if the remaining tensor is a scalar then it will return a saclar Tensor
	const T& at(const std::vector<uint64_t>& idx) const;

	// returns a Tensor containing the data in the given index
	// its just a view
	// if the remaining tensor is a scalar then it will return a saclar Tensor
	Tensor<T> operator[](const uint64_t idx) const;

	// makes a new tensor that has the same data as the old one
	Tensor<T> clone() const;

	// IF the tensor is scalar
	// returns that one element
	T& get();

	// IF the tensor is scalar
	// returns that one element
	const T& get() const;

	//& private ==========================================================================
  private:
	Tensor(const std::vector<uint64_t>& shape, const mem::Buffer data,
	       const std::vector<uint64_t>& strides, const uint64_t offset);

	void ComputeStrides();


	// base case
	template <class K>
	void getSTDVecShape(const K& k, std::vector<uint64_t>& shape);

	template <class K>
	void getSTDVecShape(const std::vector<K>& v, std::vector<uint64_t>& shape);

	// base case
	template <class K>
	void falttenSTDVec(const K& k, T* dst, uint64_t& offset);

	template <class K>
	void falttenSTDVec(const std::vector<K>& v, T* dst, uint64_t& offset);
};

} // namespace TZ