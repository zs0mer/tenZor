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
Tensor<T> Tensor<T>::operator[](const uint64_t idx) const {
	_CHECK(!indexable(), "not indexable");
	_CHECK(idx >= shape_[0], "index out of bounds");

	std::vector<uint64_t> newShape(shape_.begin() + 1, shape_.end());

	std::vector<uint64_t> newStrides(strides_.begin() + 1, strides_.end());

	uint64_t newOffset = offset_ + idx * strides_[0];

	return Tensor<T>(newShape, data_, newStrides, newOffset);
}

template <class T>
Tensor<T> Tensor<T>::clone() const {
	return Tensor(shape_, data_.clone(), strides_, offset_);
}

template <class T>
T& Tensor<T>::get() {
	_CHECK(!scalar(), "Not a scalar");
	return *static_cast<T*>(data_->data());
}

template <class T>
const T& Tensor<T>::get() const {
	_CHECK(!scalar(), "Not a scalar");
	return *static_cast<const T*>(data_->data());
}

} // namespace TZ