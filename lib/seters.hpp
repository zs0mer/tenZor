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
Tensor<T>::Tensor(const uint64_t dim, const uint64_t* shape, const uint8_t alignment,
                  mem::Allocator& allocator) {
	set(dim, shape, alignment, allocator);
}

template <class T>
template <class NestedVector>
Tensor<T> Tensor<T>::fromSTDVec(const std::vector<NestedVector>& v, const uint8_t alignment,
                                mem::Allocator& allocator) {
	Tensor<T> t;
	t.offset_ = 0;
	uint8_t currDim = 0;
	t.getSTDVecShape(v, t.shape_.data(), currDim);

	t.dim_ = static_cast<uint8_t>(currDim);

	t.computeStrides();

	uint64_t capacity = 1;
	for (int i = 0; i < t.dim_; i++)
		capacity *= t.shape_[i];


	t.data_ = mem::Buffer(capacity * sizeof(T), alignment, &allocator);

	uint64_t offset = 0;
	t.falttenSTDVec(v, t.data(), offset);
	return t;
}

template <class T>
void Tensor<T>::set(const std::vector<uint64_t>& shape, const uint8_t alignment,
                    mem::Allocator& allocator) {
	set(shape.size(), shape.begin(), alignment, allocator);
}

template <class T>
void Tensor<T>::set(const uint64_t dim, const uint64_t* shape, const uint8_t alignment,
                    mem::Allocator& allocator) {
	_CHECK(dim > MAX_DIM, "tensor dimension exceeds MAX_DIMS");
	dim_ = dim;
	offset_ = 0;

	uint64_t capacity = 1;
	for (uint8_t i = 0; i < dim_; i++) {
		shape_[i] = shape[i];
		capacity *= shape[i];
	}

	computeStrides();

	data_ = mem::Buffer(capacity * sizeof(T), alignment, &allocator);
}
} // namespace TZ