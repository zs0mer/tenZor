#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include <cstring>
#include <iostream>

#include "config.hpp"
#include "allocator.hpp"
#include "buffer.hpp"
#include "tensor.hpp"
#include "utils.hpp"


namespace TZ {

template <class T>
Tensor<T>::Tensor() : dim_(0), offset_(0), shape_({}), strides_({}), data_(nullptr) {}

template <class T>
Tensor<T>::Tensor(const std::vector<uint64_t>& shape, mem::Allocator& allocator) {
	set(shape, allocator);
}

template <class T>
Tensor<T>::Tensor(const uint8_t dim, const uint64_t* shape, mem::Allocator& allocator) {
	set(dim, shape, allocator);
}

template <class T>
Tensor<T>::Tensor(const uint8_t dim, const uint64_t* shape, const uint64_t* strides,
                  const uint64_t offset, mem::Buffer data)
    : dim_(dim), offset_(offset), data_(data) {
	for (uint8_t i = 0; i < dim; i++) {
		shape_[i] = shape[i];
		strides_[i] = strides[i];
	}
}

template <class T>
template <class NestedVector>
Tensor<T> Tensor<T>::fromSTDVec(const std::vector<NestedVector>& v, mem::Allocator& allocator) {
	Tensor<T> t;
	t.offset_ = 0;
	uint8_t currDim = 0;
	t.getSTDVecShape(v, t.shape_.data(), currDim);

	t.dim_ = static_cast<uint8_t>(currDim);

	t.computeStrides();

	uint64_t capacity = 1;
	for (int i = 0; i < t.dim_; i++)
		capacity *= t.shape_[i];


	t.data_ = mem::Buffer(capacity * sizeof(T), &allocator);

	uint64_t offset = 0;
	t.falttenSTDVec(v, t.data(), offset);
	return t;
}

template <class T>
void Tensor<T>::set(const std::vector<uint64_t>& shape, mem::Allocator& allocator) {
	set(shape.size(), shape.begin(), allocator);
}

template <class T>
void Tensor<T>::set(const uint64_t dim, const uint64_t* shape, mem::Allocator& allocator) {
	TZ_CHECK(dim > config::MAX_DIM, "tensor dimension exceeds MAX_DIMS");
	dim_ = dim;
	offset_ = 0;

	uint64_t capacity = 1;
	for (uint8_t i = 0; i < dim_; i++) {
		shape_[i] = shape[i];
		capacity *= shape[i];
	}

	computeStrides();

	data_ = mem::Buffer(capacity * sizeof(T), &allocator);
}

template <class T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& a) {
#if TZ_NORMAL_EQUAL
	if (isSameShape(*this, a) && (!dense() || offset_ != 0)) {

		apply(a, *this, [](const T& a, T& b) { b = a; });

		return *this;
	}
#endif
	return Tensor<T>(a);
}

template <class T>
Tensor<T>& Tensor<T>::operator=(Tensor<T>&& a) {
#if TZ_NORMAL_EQUAL
	if (isSameShape(*this, a)) {

		apply(a, *this, [](const T& a, T& b) { b = a; });

		return *this;
	}
#endif

	dim_ = a.dim_;
	shape_ = a.shape_;
	strides_ = a.strides_;
	offset_ = a.offset_;
	data_ = std::move(a.data_);

	return *this;
}

//&
//================================================================================================================================================================

template <class T>
uint64_t Tensor<T>::dim() const {
	return dim_;
}

template <class T>
const uint64_t* Tensor<T>::shape() const {
	return shape_.begin();
}

template <class T>
const uint64_t* Tensor<T>::_strides() const {
	return strides_.begin();
}

template <class T>
uint64_t Tensor<T>::size() const {
	if (c_size_.cached)
		return c_size_.value;

	if (dim_ == 0 && !data_->data()) {
		c_size_.set(0);
		return 0;
	}

	uint64_t elements = 1;

	for (uint8_t i = 0; i < dim_; i++)
		elements *= shape_[i];

	c_size_.set(elements);

	return elements;
};

template <class T>
T* Tensor<T>::data() {
	if (!data_->data())
		return nullptr;
	return static_cast<T*>(data_->data()) + offset_;
}

template <class T>
const T* Tensor<T>::data() const {
	if (!data_->data())
		return nullptr;
	return static_cast<const T*>(data_->data()) + offset_;
}

template <class T>
T* Tensor<T>::_rawData() {
	if (!data_->data())
		return nullptr;
	return static_cast<T*>(data_->data());
}

template <class T>
const T* Tensor<T>::_rawData() const {
	if (!data_->data())
		return nullptr;
	return static_cast<const T*>(data_->data());
}

template <class T>
uint64_t Tensor<T>::_offset() const {
	return offset_;
}

template <class T>
bool Tensor<T>::dense() const {
	if (c_dense.cached)
		return c_dense.value;


	if (empty()) {
		c_dense.set(true);
		return true;
	}

	uint64_t expected = 1;
	for (int64_t i = dim_; i-- > 0;) {
		if (strides_[i] != expected) {
			c_dense.set(false);
			return false;
		}

		expected *= shape_[i];
	}
	c_dense.set(true);

	return true;
}

template <class T>
bool Tensor<T>::scalar() const {
	return dim_ == 0 && size() == 1;
}

template <class T>
bool Tensor<T>::empty() const {
	return size() == 0;
}

template <class T>
bool Tensor<T>::indexable() const {
	return !scalar() && !empty();
}

template <class T>
mem::Allocator& Tensor<T>::allocator() const {
	return *data_->allocator();
}

template <class T>
mem::Device Tensor<T>::device() const {
	return data_->allocator()->device();
}

template <class T>
mem::Buffer Tensor<T>::_buffer() {
	return data_;
}

template <class T>
const mem::Buffer Tensor<T>::_buffer() const {
	return data_;
}


//&
//================================================================================================================================================================

template <class T>
T& Tensor<T>::at(const std::vector<uint64_t>& idx) {
	TZ_CHECK(!indexable(), "not indexable");
	TZ_CHECK(idx.size() != dim_, "incorrect number of indices");

	uint64_t linearIdx = offset_;

	for (size_t i = 0; i < dim_; ++i) {
		TZ_CHECK(idx[i] >= shape_[i], "index out of bounds");
		linearIdx += idx[i] * strides_[i];
	}

	return static_cast<T*>(data_->data())[linearIdx];
}

template <class T>
const T& Tensor<T>::at(const std::vector<uint64_t>& idx) const {
	TZ_CHECK(!indexable(), "not indexable");
	TZ_CHECK(idx.size() != dim_, "incorrect number of indices");

	uint64_t linearIdx = offset_;

	for (size_t i = 0; i < dim_; ++i) {
		TZ_CHECK(idx[i] >= shape_[i], "index out of bounds");
		linearIdx += idx[i] * strides_[i];
	}

	return static_cast<const T*>(data_->data())[linearIdx];
}

template <class T>
T& Tensor<T>::at(const uint64_t* idx) {
	uint64_t linearIdx = offset_;

	for (size_t i = 0; i < dim_; ++i) {
		TZ_CHECK(idx[i] >= shape_[i], "index out of bounds");
		linearIdx += idx[i] * strides_[i];
	}

	return static_cast<T*>(data_->data())[linearIdx];
}

template <class T>
const T& Tensor<T>::at(const uint64_t* idx) const {
	uint64_t linearIdx = offset_;

	for (size_t i = 0; i < dim_; ++i) {
		TZ_CHECK(idx[i] >= shape_[i], "index out of bounds");
		linearIdx += idx[i] * strides_[i];
	}

	return static_cast<const T*>(data_->data())[linearIdx];
}

template <class T>
Tensor<T> Tensor<T>::operator[](const uint64_t idx) {
	TZ_CHECK(!indexable(), "not indexable");
	TZ_CHECK(idx >= shape_[0], "index out of bounds");

	return Tensor<T>(dim_ - 1, &shape_[1], &strides_[1], offset_ + idx * strides_[0], data_);
}

template <class T>
const Tensor<T> Tensor<T>::operator[](const uint64_t idx) const {
	TZ_CHECK(!indexable(), "not indexable");
	TZ_CHECK(idx >= shape_[0], "index out of bounds");

	return Tensor<T>(dim_ - 1, &shape_[1], &strides_[1], offset_ + idx * strides_[0], data_);
}

template <class T>
Tensor<T> Tensor<T>::clone() const {
	Tensor<T> out(dim_, shape_.data(), *data_->allocator());

	if (dense()) {
		std::memcpy(out.data(), this->data(), size() * sizeof(T));
		return out;
	}

	apply(*this, out, [](const T& a, T& b) { b = a; });

	return out;
}

template <class T>
T& Tensor<T>::get() {
	TZ_CHECK(!scalar(), "Not a scalar");
	return *(static_cast<T*>(data_->data()) + offset_);
}

template <class T>
const T& Tensor<T>::get() const {
	TZ_CHECK(!scalar(), "Not a scalar");
	return *(static_cast<const T*>(data_->data()) + offset_);
}

//&
//================================================================================================================================================================

template <class T>
void Tensor<T>::computeStrides() {
	uint64_t k = 1;

	for (int64_t i = dim_; i-- > 0;) {
		strides_[i] = k;
		k *= shape_[i];
	}
}


template <class T>
template <class K>
void Tensor<T>::getSTDVecShape(const K& k, uint64_t* const shape, uint8_t& currDim) {
	return;
}

template <class T>
template <class K>
void Tensor<T>::getSTDVecShape(const std::vector<K>& v, uint64_t* const shape, uint8_t& currDim) {
	shape[currDim++] = v.size();
	TZ_CHECK(currDim > config::MAX_DIM, "tensor dimension exceeds MAX_DIMS");
	if (v.empty())
		return;
	getSTDVecShape(v[0], shape, currDim);
}


template <class T>
template <class K>
void Tensor<T>::falttenSTDVec(const K& k, T* dst, uint64_t& offset) {
	dst[offset++] = static_cast<T>(k);
}

template <class T>
template <class K>
void Tensor<T>::falttenSTDVec(const std::vector<K>& v, T* dst, uint64_t& offset) {
	for (const auto& i : v)
		falttenSTDVec(i, dst, offset);
}

template <class T>
bool Tensor<T>::isSameShape(const Tensor<T>& a, const Tensor<T>& b) {
	if (a.dim_ != b.dim_)
		return false;

	for (uint8_t i = 0; i < a.dim_; i++)
		if (a.shape()[i] != b.shape()[i])
			return false;

	return true;
}


template <class T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& t) {
	if (t.empty()) {
		os << "[]";
		return os;
	}

	if (t.scalar()) {
		os << t.get();
		return os;
	}

	if (t.dim() == 1) {
		os << "\n[";
		for (int i = 0; i < t.size() - 1; i++)
			os << t[i].get() << ", ";

		os << t[t.size() - 1].get() << "]";
		return os;
	}

	os << "[";
	for (int i = 0; i < t.shape()[0] - 1; i++)
		os << t[i] << ",";
	os << t[t.shape()[0] - 1] << "\n]";
	return os;
}

template <typename Derived, typename T>
std::ostream& operator<<(std::ostream& os, const _tensorWrapper<Derived, T>& t) {
	os << t.tensor();
	return os;
}

} // namespace TZ
