#pragma once

namespace TZ {

template <class T> class Tensor {
  private:
	std::vector<uint64_t> shape_;
	std::vector<uint64_t> strides_;
	uint64_t offset_ = 0;
	mem::Buffer data_;

	void compute_default_strides();

  public:
	//& seters ===========================================================================

	Tensor(const std::vector<uint64_t>& shape, const uint8_t alignment = 64,
	       mem::Allocator& allocator = mem::salloc::instance());

	template <class nestedVector>
	Tensor(const std::vector<nestedVector>& vec, const uint8_t alignment = 64,
	       mem::Allocator& allocator = mem::salloc::instance());

	template <class nestedList>
	Tensor(const std::initializer_list<nestedList> vec, const uint8_t alignment = 64,
	       mem::Allocator& allocator = mem::salloc::instance());

	Tensor(const std::vector<uint64_t>& shape_, const mem::Buffer data_,
	       const std::vector<uint64_t>& strides_, const uint64_t offset_);

	Tensor(const Tensor& other);

	Tensor& operator=(const Tensor& other);

	Tensor(Tensor&& other);

	Tensor& operator=(Tensor&& other);

	//& metadata geters ==================================================================

	size_t dim() const;

	const std::vector<uint64_t>& shape() const;

	T* data();

	const T* data() const;

	bool empty() const;

	bool is_contiguous() const;

	bool isFullBuffer() const;

	bool isScalar() const;

	//& geters ===========================================================================

	T& at(const std::vector<uint64_t>& idx);

	const T& at(const std::vector<uint64_t>& idx) const;

	Tensor<T> operator[](const uint64_t idx) const;

	Tensor<T> clone() const;

	T scalarVal() const;
};

} // namespace TZ