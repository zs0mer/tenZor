#pragma once
// only the declarations are in this file


namespace TZ {

template <class T>
// standard tensor class
class Tensor {
  private:
	uint8_t dim_;
	std::array<uint64_t, MAX_DIM> shape_;
	std::array<uint64_t, MAX_DIM> strides_;
	uint64_t offset_;
	mem::Buffer data_;

	// the tensor can be:
	// - Normal - dimenson: anything   - shape: anything else               - data: anything
	// - Scalar - dimenson: 0          - shape: {}                          - data: sizeof(T)
	// - Empty  - dimenson: 0          - shape: has at least one 0 in it    - data: nullptr
	// - Null   - dimenson: 0          - shape: {}                          - data: nullptr

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
	const uint64_t* shape() const;

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

	// returns true if this is not only a part of a bigger tensor
	// it means that the tensor is layed out flat in memory
	// if softCheck = true it can return true even if
	// the data is not at the start of the memory pointer, it is offseted by offset()
	bool isContiguous(const bool softCheck = false) const;

	// returns true if the tensor is a scalar
	bool scalar() const;

	// returns true if the tensor is Empty or Null
	// returns size() == 0
	bool empty() const;

	// returns true if the tensor is normal
	bool indexable() const;

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
	void getSTDVecShape(const K& k, std::vector<uint64_t>& shape, uint8_t& currDim);

	template <class K>
	void getSTDVecShape(const std::vector<K>& v, std::vector<uint64_t>& shape, uint8_t& currDim);

	// base case
	template <class K>
	void falttenSTDVec(const K& k, T* dst, uint64_t& offset);

	template <class K>
	void falttenSTDVec(const std::vector<K>& v, T* dst, uint64_t& offset);
};

} // namespace TZ