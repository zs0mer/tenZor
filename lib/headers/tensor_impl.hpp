#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "allocator.hpp"
#include "buffer.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include "kernel_functions.hpp"


namespace TZ::impl {

// # -------------------------
#ifdef TZ_NORMAL_EQUAL
// nothing
#else
#define TZ_NORMAL_EQUAL 1
#endif
// # -------------------------


template <class T>
TensorIMPL<T>::TensorIMPL()
    : dim_(0), offset_(0), shape_({}), strides_({}), data_(nullptr), size_(0), dense_(true),
      broadcasted_(false) {}

template <class T>
TensorIMPL<T>::TensorIMPL(const std::initializer_list<uint64_t>& shape, Device device) {
	set(shape, device);
}

template <class T>
TensorIMPL<T>::TensorIMPL(const uint8_t dim, const uint64_t* shape, Device device) {
	set(dim, shape, device);
}

template <class T>
TensorIMPL<T>::TensorIMPL(const uint8_t dim, const uint64_t* shape, const uint64_t* strides,
                          const uint64_t offset, mem::Buffer data) {
	set(dim, shape, strides, offset, data);
}

template <class T>
void TensorIMPL<T>::set(const std::initializer_list<uint64_t>& shape, Device device) {
	set(shape.size(), shape.begin(), device);
}

template <class T>
void TensorIMPL<T>::set(const uint64_t dim, const uint64_t* shape, Device device) {
	TZ_CHECK(dim <= MAX_DIM, "tensor dimension exceeds MAX_DIMS");
	dim_ = dim;
	offset_ = 0;

	uint64_t capacity = 1;
	for (uint8_t i = 0; i < dim_; i++) {
		shape_[i] = shape[i];
		capacity *= shape[i];
	}


	data_ = mem::Buffer(capacity * sizeof(T), &mem::defaultAllocator(device));
	computeStrides();
	computeMetadata();
}

template <class T>
void TensorIMPL<T>::set(const uint8_t dim, const uint64_t* shape, const uint64_t* strides,
                        const uint64_t offset, mem::Buffer data) {
	dim_ = dim;
	offset_ = offset;
	data_ = data;
	for (uint8_t i = 0; i < dim; i++) {
		shape_[i] = shape[i];
		strides_[i] = strides[i];
	}
	computeMetadata();
}

template <class T>
TensorIMPL<T>& TensorIMPL<T>::operator=(const TensorIMPL<T>& a) {
#if TZ_NORMAL_EQUAL
	if (isSameShape(*this, a) && a.device() == this->device()) {

		apply(a, *this, Copy<T>{});

		return *this;
	}
#endif

	dim_ = a.dim_;
	shape_ = a.shape_;
	strides_ = a.strides_;
	offset_ = a.offset_;
	data_ = a.data_;

	return *this;
}

template <class T>
TensorIMPL<T>& TensorIMPL<T>::operator=(TensorIMPL<T>&& a) {
#if TZ_NORMAL_EQUAL
	if (isSameShape(*this, a) && a.device() == this->device()) {

		apply(a, *this, Copy<T>{});

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


// # ====================================================================================

template <class T>
uint64_t TensorIMPL<T>::dim() const {
	return dim_;
}

template <class T>
const uint64_t* TensorIMPL<T>::shape() const {
	return shape_.begin();
}

template <class T>
const uint64_t* TensorIMPL<T>::strides() const {
	return strides_.begin();
}

template <class T>
uint64_t TensorIMPL<T>::size() const {
	return size_;
};

template <class T>
T* TensorIMPL<T>::data() {
	if (!data_->data())
		return nullptr;
	return static_cast<T*>(data_->data()) + offset_;
}

template <class T>
const T* TensorIMPL<T>::data() const {
	if (!data_->data())
		return nullptr;
	return static_cast<const T*>(data_->data()) + offset_;
}

template <class T>
T* TensorIMPL<T>::rawData() {
	if (!data_->data())
		return nullptr;
	return static_cast<T*>(data_->data());
}

template <class T>
const T* TensorIMPL<T>::rawData() const {
	if (!data_->data())
		return nullptr;
	return static_cast<const T*>(data_->data());
}

template <class T>
uint64_t TensorIMPL<T>::offset() const {
	return offset_;
}

template <class T>
bool TensorIMPL<T>::dense() const {
	return dense_;
}

template <class T>
bool TensorIMPL<T>::scalar() const {
	return dim_ == 0 && size() == 1;
}

template <class T>
bool TensorIMPL<T>::empty() const {
	return size() == 0;
}

template <class T>
bool TensorIMPL<T>::indexable() const {
	return !scalar() && !empty();
}

template <class T>
bool TensorIMPL<T>::broadcasted() const {
	return broadcasted_;
}


template <class T>
mem::Allocator& TensorIMPL<T>::allocator() const {
	return *data_->allocator();
}

template <class T>
Device TensorIMPL<T>::device() const {
	return data_->allocator()->device();
}

template <class T>
mem::Buffer TensorIMPL<T>::buffer() {
	return data_;
}

template <class T>
const mem::Buffer TensorIMPL<T>::buffer() const {
	return data_;
}


// # ====================================================================================

template <class T>
T& TensorIMPL<T>::at(const std::initializer_list<uint64_t>& idx) {
#if TZ_UNMUTABLE_BRODCASTS
	TZ_CHECK(!broadcasted(), "can't use at() on broadcasted tensors");
#endif
	TZ_CHECK(device() == CPU, "not on the CPU");
	TZ_CHECK(indexable(), "not indexable");
	TZ_CHECK(idx.size() == dim_, "incorrect number of indices");

	uint64_t linearIdx = offset_;

	for (size_t i = 0; i < dim_; i++) {
		TZ_CHECK(idx.begin()[i] < shape_[i], "index out of bounds");
		linearIdx += idx.begin()[i] * strides_[i];
	}

	return static_cast<T*>(data_->data())[linearIdx];
}

template <class T>
const T& TensorIMPL<T>::at(const std::initializer_list<uint64_t>& idx) const {
#if TZ_UNMUTABLE_BRODCASTS
	TZ_CHECK(!broadcasted(), "can't use at() on broadcasted tensors");
#endif
	TZ_CHECK(device() == CPU, "not on the CPU");
	TZ_CHECK(indexable(), "not indexable");
	TZ_CHECK(idx.size() == dim_, "incorrect number of indices");

	uint64_t linearIdx = offset_;

	for (size_t i = 0; i < dim_; i++) {
		TZ_CHECK(idx.begin()[i] < shape_[i], "index out of bounds");
		linearIdx += idx.begin()[i] * strides_[i];
	}

	return static_cast<const T*>(data_->data())[linearIdx];
}

template <class T>
T& TensorIMPL<T>::at(const uint64_t* idx) {
#if TZ_UNMUTABLE_BRODCASTS
	TZ_CHECK(!broadcasted(), "can't use at() on broadcasted tensors");
#endif
	TZ_CHECK(device() == CPU, "not on the CPU");
	uint64_t linearIdx = offset_;

	for (size_t i = 0; i < dim_; i++) {
		TZ_CHECK(idx[i] < shape_[i], "index out of bounds");
		linearIdx += idx[i] * strides_[i];
	}

	return static_cast<T*>(data_->data())[linearIdx];
}

template <class T>
const T& TensorIMPL<T>::at(const uint64_t* idx) const {
#if TZ_UNMUTABLE_BRODCASTS
	TZ_CHECK(!broadcasted(), "can't use at() on broadcasted tensors");
#endif
	TZ_CHECK(device() == CPU, "not on the CPU");
	uint64_t linearIdx = offset_;

	for (size_t i = 0; i < dim_; i++) {
		TZ_CHECK(idx[i] < shape_[i], "index out of bounds");
		linearIdx += idx[i] * strides_[i];
	}

	return static_cast<const T*>(data_->data())[linearIdx];
}

template <class T>
TensorIMPL<T> TensorIMPL<T>::operator[](const uint64_t idx) {
#if TZ_UNMUTABLE_BRODCASTS
	TZ_CHECK(!broadcasted(), "can't use operator[] on broadcasted tensors");
#endif
	TZ_CHECK(indexable(), "not indexable");
	TZ_CHECK(idx < shape_[0], "index out of bounds");

	return TensorIMPL<T>(dim_ - 1, &shape_[1], &strides_[1], offset_ + idx * strides_[0], data_);
}

template <class T>
TensorIMPL<T> TensorIMPL<T>::operator[](const uint64_t idx) const {
#if TZ_UNMUTABLE_BRODCASTS
	TZ_CHECK(!broadcasted(), "can't use operator[] on broadcasted tensors");
#endif
	TZ_CHECK(indexable(), "not indexable");
	TZ_CHECK(idx < shape_[0], "index out of bounds");

	return TensorIMPL<T>(dim_ - 1, &shape_[1], &strides_[1], offset_ + idx * strides_[0], data_);
}

template <class T>
TensorIMPL<T> TensorIMPL<T>::clone() const {
	TensorIMPL<T> out(dim_, shape_.data(), device());

	if (dense()) {
		if (device() == CPU)
			std::memcpy(out.data(), this->data(), size() * sizeof(T));
		if (device() == GPU)
			cuda::memCopyOnGPU(out.data(), this->data(), size() * sizeof(T));
		return out;
	}

	apply(*this, out, Copy<T>{});

	return out;
}

template <class T>
TensorIMPL<T> TensorIMPL<T>::copyTo(Device toDevice) const {
	if (device() == toDevice)
		return clone();

	TensorIMPL<T> t = *this;
	if (!dense())
		t = t.clone();

	mem::Buffer buff = mem::Buffer(t.data_->size(), &mem::defaultAllocator(toDevice));
	uint64_t byteSize = t.size() * sizeof(T);

	if (toDevice == GPU)
		cuda::copyToGPU(buff->data(), t.data(), byteSize);
	else if (toDevice == CPU)
		cuda::copyToCPU(buff->data(), t.data(), byteSize);


	return TensorIMPL<T>(dim_, t.shape_.data(), t.strides_.data(), 0, buff);
}

template <class T>
T& TensorIMPL<T>::get() {
	TZ_CHECK(device() == CPU, "not on the CPU");
	TZ_CHECK(scalar(), "Not a scalar");
	return *(static_cast<T*>(data_->data()) + offset_);
}

template <class T>
const T& TensorIMPL<T>::get() const {
	TZ_CHECK(device() == CPU, "not on the CPU");
	TZ_CHECK(scalar(), "Not a scalar");
	return *(static_cast<const T*>(data_->data()) + offset_);
}


// # ====================================================================================

template <class T>
void TensorIMPL<T>::computeStrides() {
	uint64_t k = 1;

	for (int64_t i = dim_; i-- > 0;) {
		strides_[i] = k;
		k *= shape_[i];
	}
}

template <class T>
void TensorIMPL<T>::computeMetadata() {

	if (!data_->data()) {
		size_ = 0;
		dense_ = true;
		broadcasted_ = false;
		return;
	}

	uint64_t elements = 1;
	broadcasted_ = false;

	for (uint8_t i = 0; i < dim_; i++) {
		elements *= shape_[i];
		if (strides_[i] == 0)
			broadcasted_ = true;
	}

	size_ = elements;

	uint64_t expected = 1;
	for (int64_t i = dim_; i-- > 0;) {
		if (strides_[i] != expected) {
			dense_ = false;
			return;
		}

		expected *= shape_[i];
	}
	dense_ = true;
}


template <class T>
bool TensorIMPL<T>::isSameShape(const TensorIMPL<T>& a, const TensorIMPL<T>& b) {
	if (a.dim_ != b.dim_)
		return false;

	for (uint8_t i = 0; i < a.dim_; i++)
		if (a.shape()[i] != b.shape()[i])
			return false;

	return true;
}

template <class T>
cuda::SimpleTensor<T> TensorIMPL<T>::getCudaTensor() {
	cuda::SimpleTensor<T> out;
	out.dim = static_cast<uint8_t>(dim_);
	out.offset = offset_;
	out.data = reinterpret_cast<T*>(data_->data());
	out.size = this->size();
	out.dense = this->dense();

	// Just copy the small metadata directly into the struct
	for (uint8_t i = 0; i < dim_; i++) {
		out.shape[i] = shape_[i];
		out.strides[i] = strides_[i];
	}

	return out;
}

template <class T>
const cuda::SimpleTensor<T> TensorIMPL<T>::getCudaTensor() const {
	cuda::SimpleTensor<T> out;
	out.dim = static_cast<uint8_t>(dim_);
	out.offset = offset_;
	out.data = const_cast<T*>(reinterpret_cast<const T*>(data_->data()));
	out.size = this->size();
	out.dense = this->dense();

	// Just copy the small metadata directly into the struct
	for (uint8_t i = 0; i < dim_; i++) {
		out.shape[i] = shape_[i];
		out.strides[i] = strides_[i];
	}

	return out;
}

template <class T>
std::ostream& operator<<(std::ostream& os, const TensorIMPL<T>& t) {
	if (t.empty()) {
		os << "[]";
		return os;
	}

	if (t.scalar()) {
		os << t.get();
		return os;
	}

	if (t.size() > 1000)
		return os << "[To many elements to print: " << t.size() << " elements]\n";

	if (t.dim() == 1) {
		os << "\n[";
		for (uint64_t i = 0; i < t.size() - 1; i++)
			os << t[i].get() << ", ";

		os << t[t.size() - 1].get() << "]";
		return os;
	}

	os << "[";
	for (uint64_t i = 0; i < t.shape()[0] - 1; i++)
		os << t[i] << ",";
	os << t[t.shape()[0] - 1] << "\n]";
	return os;
}

} // namespace TZ::impl
