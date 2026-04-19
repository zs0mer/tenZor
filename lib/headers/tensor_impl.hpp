#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include <cstring>
#include <iostream>

#include "allocator.hpp"
#include "buffer.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include "kernel_functions.hpp"


namespace TZ::internal {

// # -------------------------
#ifdef TZ_NORMAL_EQUAL
// nothing
#else
#define TZ_NORMAL_EQUAL 1
#endif
// # -------------------------


template <class T>
TensorIMPL<T>::TensorIMPL() : dim_(0), offset_(0), shape_({}), strides_({}), data_(nullptr) {}

template <class T>
TensorIMPL<T>::TensorIMPL(const std::vector<uint64_t>& shape, Device device) {
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
void TensorIMPL<T>::set(const std::vector<uint64_t>& shape, Device device) {
	set(shape.size(), shape.data(), device);
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

	computeStrides();

	data_ = mem::Buffer(capacity * sizeof(T), &mem::defaultAllocator(device));
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
	if (c_size_.isCached())
		return c_size_.get();

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
	if (c_dense_.isCached())
		return c_dense_.get();


	if (empty()) {
		c_dense_.set(true);
		return true;
	}

	uint64_t expected = 1;
	for (int64_t i = dim_; i-- > 0;) {
		if (strides_[i] != expected) {
			c_dense_.set(false);
			return false;
		}

		expected *= shape_[i];
	}
	c_dense_.set(true);

	return true;
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
T& TensorIMPL<T>::at(const std::vector<uint64_t>& idx) {
	TZ_CHECK(device() == CPU, "not on the CPU");
	TZ_CHECK(indexable(), "not indexable");
	TZ_CHECK(idx.size() == dim_, "incorrect number of indices");

	uint64_t linearIdx = offset_;

	for (size_t i = 0; i < dim_; ++i) {
		TZ_CHECK(idx[i] < shape_[i], "index out of bounds");
		linearIdx += idx[i] * strides_[i];
	}

	return static_cast<T*>(data_->data())[linearIdx];
}

template <class T>
const T& TensorIMPL<T>::at(const std::vector<uint64_t>& idx) const {
	TZ_CHECK(device() == CPU, "not on the CPU");
	TZ_CHECK(indexable(), "not indexable");
	TZ_CHECK(idx.size() == dim_, "incorrect number of indices");

	uint64_t linearIdx = offset_;

	for (size_t i = 0; i < dim_; ++i) {
		TZ_CHECK(idx[i] < shape_[i], "index out of bounds");
		linearIdx += idx[i] * strides_[i];
	}

	return static_cast<const T*>(data_->data())[linearIdx];
}

template <class T>
T& TensorIMPL<T>::at(const uint64_t* idx) {
	TZ_CHECK(device() == CPU, "not on the CPU");
	uint64_t linearIdx = offset_;

	for (size_t i = 0; i < dim_; ++i) {
		TZ_CHECK(idx[i] < shape_[i], "index out of bounds");
		linearIdx += idx[i] * strides_[i];
	}

	return static_cast<T*>(data_->data())[linearIdx];
}

template <class T>
const T& TensorIMPL<T>::at(const uint64_t* idx) const {
	TZ_CHECK(device() == CPU, "not on the CPU");
	uint64_t linearIdx = offset_;

	for (size_t i = 0; i < dim_; ++i) {
		TZ_CHECK(idx[i] < shape_[i], "index out of bounds");
		linearIdx += idx[i] * strides_[i];
	}

	return static_cast<const T*>(data_->data())[linearIdx];
}

template <class T>
TensorIMPL<T> TensorIMPL<T>::operator[](const uint64_t idx) {
	TZ_CHECK(indexable(), "not indexable");
	TZ_CHECK(idx < shape_[0], "index out of bounds");

	return TensorIMPL<T>(dim_ - 1, &shape_[1], &strides_[1], offset_ + idx * strides_[0], data_);
}

template <class T>
const TensorIMPL<T> TensorIMPL<T>::operator[](const uint64_t idx) const {
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

	mem::Buffer buff;
	if (toDevice == GPU) {
		buff = mem::Buffer(t.data_->size(), &mem::defaultAllocator(toDevice));
		cuda::copyToGPU(buff->data(), t.data(), t.data_->size());
	} else if (toDevice == CPU) {
		buff = mem::Buffer(t.data_->size(), &mem::defaultAllocator(toDevice));
		cuda::copyToCPU(buff->data(), t.data(), t.data_->size());
	}

	return TensorIMPL<T>(dim_, shape_.data(), strides_.data(), 0, buff);
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
void TensorIMPL<T>::gpuMetadataLazyInit() {
	if (device() != GPU || gpuMetadata_->data() || dim_ == 0)
		return;

	gpuMetadata_ = mem::Buffer(2 * dim_ * sizeof(uint64_t), &mem::defaultAllocator(GPU));

	TZ_CHECK_(gpuMetadata_->data());
	cuda::copyToGPU(gpuMetadata_->data(), shape_.data(), dim_ * sizeof(uint64_t));
	cuda::copyToGPU(reinterpret_cast<uint64_t*>(gpuMetadata_->data()) + dim_, strides_.data(),
	                dim_ * sizeof(uint64_t));
}

template <class T>
void TensorIMPL<T>::computeStrides() {
	uint64_t k = 1;

	for (int64_t i = dim_; i-- > 0;) {
		strides_[i] = k;
		k *= shape_[i];
	}
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
const cuda::SimpleTensor<T> TensorIMPL<T>::getCudaTensor(TensorIMPL<T> t) {
	t.gpuMetadataLazyInit();
	return cuda::SimpleTensor<T>(
	    {.dim = static_cast<uint8_t>(t.dim()),
	     .shape = reinterpret_cast<uint64_t*>(t.gpuMetadata_->data()),
	     .strides = reinterpret_cast<uint64_t*>(t.gpuMetadata_->data()) + t.dim_,
	     .offset = t.offset(),
	     .data = reinterpret_cast<T*>(t.rawData()),
	     .size = t.size(),
	     .dense = t.dense()});
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

} // namespace TZ::internal
