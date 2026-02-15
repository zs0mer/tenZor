#pragma once

namespace TZ {

template <class T>
Tensor<T>::Tensor(const std::vector<uint64_t>& shape, const uint8_t alignment,
                  mem::Allocator& allocator) {
	set(shape, alignment, allocator);
}

template <class T>
template <class NestedVector>
Tensor<T>::Tensor(const std::vector<NestedVector>& v, const uint8_t alignment,
                  mem::Allocator& allocator)
    : offset_(0) {

	getSTDVecShape(v, shape_);
	ComputeStrides();

	uint64_t capacity = 1;
	for (uint64_t i : shape_) {
		capacity *= i;
		if (i == 0) {
			shape_ = {};
			strides_ = {};
			capacity = 1;
			break;
		}
	}

	data_ = mem::Buffer(capacity * sizeof(T), alignment, &allocator);

	uint64_t offset = 0;
	falttenSTDVec(v, data(), offset);
}

template <class T>
void Tensor<T>::set(const std::vector<uint64_t>& shape, const uint8_t alignment,
                    mem::Allocator& allocator) {
	shape_ = shape;
	offset_ = 0;
	ComputeStrides();

	uint64_t capacity = 1;
	for (uint64_t i : shape_) {
		capacity *= i;
		if (i == 0) {
			shape_ = {};
			strides_ = {};
			capacity = 1;
			break;
		}
	}

	data_ = mem::Buffer(capacity * sizeof(T), alignment, &allocator);
}

template <class T>
template <class NestedVector>
Tensor<T>& Tensor<T>::operator=(const std::vector<NestedVector>& v) {
	*this = Tensor<T>(v);
	return *this;
}
} // namespace TZ