#pragma once
//~ only the declarations are in this file

namespace TZ {

template <class T>
class Tensor {
  private:
	std::vector<uint64_t> shape_;
	std::vector<uint64_t> strides_;
	uint64_t offset_;
	mem::Buffer data_;

  public:
	//& seters ===========================================================================

	Tensor() = default;

	Tensor(const std::vector<uint64_t>& shape, const uint8_t alignment = 64,
	       mem::Allocator& allocator = mem::salloc::instance());

	Tensor(const std::vector<uint64_t>& shape, const mem::Buffer data,
	       const std::vector<uint64_t>& strides, const uint64_t offset);

	template <class nestedVector>
	Tensor(const std::vector<nestedVector>& v, const uint8_t alignment = 64,
	       mem::Allocator& allocator = mem::salloc::instance());

	template <class nestedList>
	Tensor(const std::initializer_list<nestedList> list, const uint8_t alignment = 64,
	       mem::Allocator& allocator = mem::salloc::instance());

	void set(const std::vector<uint64_t>& shape, const uint8_t alignment = 64,
	         mem::Allocator& allocator = mem::salloc::instance());

	template <class nestedVector> //
	Tensor& operator=(const std::vector<nestedVector>& v);

	template <class nestedList> //
	Tensor& operator=(const std::initializer_list<nestedList> list);


	Tensor(const Tensor&) = default;

	Tensor& operator=(const Tensor&) = default;

	Tensor(Tensor&&) = default;

	Tensor& operator=(Tensor&&) = default;

	//& metadata geters ==================================================================

	uint64_t dim() const;

	const std::vector<uint64_t>& shape() const;

	const uint64_t numel() const;

	uint64_t offset() const;

	T* data();

	const T* data() const;

	bool empty() const;

	bool isContiguous(bool softCheck = false) const;

	bool isScalar() const;

	//& geters ===========================================================================

	T& at(const std::vector<uint64_t>& idx);

	const T& at(const std::vector<uint64_t>& idx) const;

	Tensor<T> operator[](const uint64_t idx) const;

	Tensor<T> clone() const;

	T& scalarVal();

	const T& scalarVal() const;

  private:
	void ComputeStrides();

	template <class nestedVector>
	void getSTDVecShape(const nestedVector& v, std::vector<uint64_t>& shape);

	template <class nestedVector>
	void falttenSTDVec(const nestedVector& v, T* dst, uint64_t& offset);

	template <class V>
	auto ilistToSTDVector(std::initializer_list<V> list);
};

} // namespace TZ