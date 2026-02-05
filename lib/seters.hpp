#pragma once

namespace TZ {

template <class T>
Tensor<T>::Tensor(const std::vector<uint64_t>& shape, const uint8_t alignment,
                  mem::Allocator& allocator)
    : shape_(shape), offset_(0) {
	compute_default_strides();

	uint64_t capacity = 1;
	for (uint64_t i : shape_)
		capacity *= i;

	buffer_ = mem::Buffer(capacity * sizeof(T), alignment, &allocator);
}

template <class T>
Tensor<T>::Tensor(const std::vector<uint64_t>& shape, const mem::Buffer data,
                  const std::vector<uint64_t>& strides, const uint64_t offset)
    : shape_(shape), data_(data), strides_(strides), offset_(offset) {}


template <class T>
template <class nestedVector>
Tensor<T>::Tensor(const std::vector<nestedVector>& vec, const uint8_t alignment,
                  mem::Allocator& allocator) {}

template <class T>
template <class nestedList>
Tensor<T>::Tensor(const std::initializer_list<nestedList> vec, const uint8_t alignment,
                  mem::Allocator& allocator) {}

} // namespace TZ