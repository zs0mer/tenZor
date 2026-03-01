#pragma once

namespace TZ {

template <class T>
T& Tensor<T>::at(const std::vector<uint64_t>& idx) {
	_CHECK(!indexable(), "not indexable");
	_CHECK(idx.size() != dim_, "incorrect number of indices");

	uint64_t linearIdx = offset_;

	for (size_t i = 0; i < dim_; ++i) {
		_CHECK(idx[i] >= shape_[i], "index out of bounds");
		linearIdx += idx[i] * strides_[i];
	}

	return static_cast<T*>(data_->data())[linearIdx];
}

template <class T>
const T& Tensor<T>::at(const std::vector<uint64_t>& idx) const {
	_CHECK(!indexable(), "not indexable");
	_CHECK(idx.size() != dim_, "incorrect number of indices");

	uint64_t linearIdx = offset_;

	for (size_t i = 0; i < dim_; ++i) {
		_CHECK(idx[i] >= shape_[i], "index out of bounds");
		linearIdx += idx[i] * strides_[i];
	}

	return static_cast<const T*>(data_->data())[linearIdx];
}

template <class T>
T& Tensor<T>::at(const uint64_t* const idx) {
	_CHECK(!indexable(), "not indexable");

	uint64_t linearIdx = offset_;

	for (size_t i = 0; i < dim_; ++i) {
		_CHECK(idx[i] >= shape_[i], "index out of bounds");
		linearIdx += idx[i] * strides_[i];
	}

	return static_cast<T*>(data_->data())[linearIdx];
}

template <class T>
const T& Tensor<T>::at(const uint64_t* const idx) const {
	_CHECK(!indexable(), "not indexable");

	uint64_t linearIdx = offset_;

	for (size_t i = 0; i < dim_; ++i) {
		_CHECK(idx[i] >= shape_[i], "index out of bounds");
		linearIdx += idx[i] * strides_[i];
	}


	return static_cast<const T*>(data_->data())[linearIdx];
}

template <class T>
Tensor<T> Tensor<T>::operator[](const uint64_t idx) const {
	_CHECK(!indexable(), "not indexable");
	_CHECK(idx >= shape_[0], "index out of bounds");

	Tensor<T> result = *this;
	result.offset_ = offset_ + idx * strides_[0];

	// shift metadata left
	for (uint32_t i = 1; i < dim_; ++i) {
		result.shape_[i - 1] = result.shape_[i];
		result.strides_[i - 1] = result.strides_[i];
	}

	result.dim_--;
	result.c_isDenseCached_ = false;
	result.c_sizeCached_ = false;

	return result;
}

template <class T>
Tensor<T> Tensor<T>::clone() const {
	return Tensor(shape_, data_.clone(), strides_, offset_);
}

template <class T>
T& Tensor<T>::get() {
	_CHECK(!scalar(), "Not a scalar");
	return *(static_cast<T*>(data_->data()) + offset_);
}

template <class T>
const T& Tensor<T>::get() const {
	_CHECK(!scalar(), "Not a scalar");
	return *(static_cast<const T*>(data_->data()) + offset_);
}

} // namespace TZ