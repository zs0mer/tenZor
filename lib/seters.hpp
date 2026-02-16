#pragma once

namespace TZ {

template <class T>
Tensor<T>::Tensor() : dim_(0), offset_(0), shape_({}), strides_({}), data_(nullptr) {}

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
	uint8_t currDim = 0;
	getSTDVecShape(v, shape_, currDim);

	_CHECK(shape.size() > MAX_DIM, "tensor dimension exceeds MAX_DIMS");
	dim_ = static_cast<int8_t>(shape.size());

	ComputeStrides();

	uint64_t capacity = 1;
	for (int i = 0; i < dim_; i++)
		capacity *= shape_[i];


	data_ = mem::Buffer(capacity * sizeof(T), alignment, &allocator);

	uint64_t offset = 0;
	falttenSTDVec(v, data(), offset);
}

template <class T>
void Tensor<T>::set(const std::vector<uint64_t>& shape, const uint8_t alignment,
                    mem::Allocator& allocator) {
	_CHECK(shape.size() > MAX_DIM, "tensor dimension exceeds MAX_DIMS");
	dim_ = shape.size();
	offset_ = 0;

	uint64_t capacity = 1;
	for (uint8_t i = 0; i < dim_; i++) {
		shape_[i] = shape[i];
		capacity *= shape[i];
	}

	ComputeStrides();

	data_ = mem::Buffer(capacity * sizeof(T), alignment, &allocator);
}
} // namespace TZ