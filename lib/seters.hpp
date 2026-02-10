#pragma once

namespace TZ {

template <class T>
Tensor<T>::Tensor(const std::vector<uint64_t>& shape, const uint8_t alignment,
                  mem::Allocator& allocator) {
	set(shape, alignment, allocator);
}

template <class T>
Tensor<T>::Tensor(const std::vector<uint64_t>& shape, const mem::Buffer data,
                  const std::vector<uint64_t>& strides, const uint64_t offset)
    : shape_(shape), data_(data), strides_(strides), offset_(offset) {}


template <class T>
template <class nestedVector>
void Tensor<T>::getSTDVecShape(const nestedVector& v, std::vector<uint64_t>& shape) {
	shape.push_back(v.size());

	if constexpr (!isSTDVector<typename nestedVector::value_type>)
		return;
	if (v.empty())
		return;

	const auto& first = v.front();
	for (const auto& sub : v)
		_CHECK(sub.size() != first.size(), "non-rectangular nested vector construction")

	getSTDVecShape(first, shape);
}

template <class T>
template <class nestedVector>
void Tensor<T>::falttenSTDVec(const nestedVector& v, T* dst, uint64_t& offset) {
	if constexpr (isSTDVector<typename nestedVector::value_type>)
		for (const auto& sub : v)
			falttenSTDVec(sub, dst, offset);
	else
		for (const auto& x : v)
			dst[offset++] = x;
}

template <class T>
template <class nestedVector>
Tensor<T>::Tensor(const std::vector<nestedVector>& v, const uint8_t alignment,
                  mem::Allocator& allocator)
    : offset_(0) {

	getSTDVecShape(v, shape_);
	ComputeStrides();

	uint64_t capacity = 1;
	for (uint64_t s : shape_)
		capacity *= s;

	data_ = mem::Buffer(capacity * sizeof(T), alignment, &allocator);

	uint64_t offset = 0;
	flatten(v, data(), offset);
}


template <class T>
template <class V>
auto Tensor<T>::ilistToSTDVector(std::initializer_list<V> list) {
	if constexpr (!isIlist<V>)
		return std::vector<V>(list);

	_CHECK(list.size() == 0, "Tensor initializer_list cannot be empty");
	std::vector<decltype(ilistToSTDVector(*list.begin()))> out;
	out.reserve(list.size());

	for (const auto& sub : list)
		out.push_back(ilistToSTDVector(sub));

	return out;
}

template <class T>
template <class nestedList>
Tensor<T>::Tensor(const std::initializer_list<nestedList> list, const uint8_t alignment,
                  mem::Allocator& allocator)
    : Tensor(ilistToSTDVector(list), alignment, allocator) {}


template <class T>
void Tensor<T>::set(const std::vector<uint64_t>& shape, const uint8_t alignment = 64,
                    mem::Allocator& allocator = mem::salloc::instance()) {
	shape_ = shape;
	offset_ = 0;
	ComputeStrides();

	uint64_t capacity = 1;
	for (uint64_t i : shape_) {
		capacity *= i;
		if (i == 0) {
			shape_ = {};
			strides_;
			capacity = 1;
			break;
		}
	}

	buffer_ = mem::Buffer(capacity * sizeof(T), alignment, &allocator);
}

template <class T>
template <class nestedVector>
Tensor<T>& Tensor<T>::operator=(const std::vector<nestedVector>& v) {
	*this = Tensor<T>(v);
	return *this;
}

template <class T>
template <class nestedList>
Tensor<T>& Tensor<T>::operator=(const std::initializer_list<nestedList> list) {
	*this = Tensor<T>(list);
	return *this;
}

} // namespace TZ